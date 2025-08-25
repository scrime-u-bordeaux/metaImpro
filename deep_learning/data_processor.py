import os
from music21 import converter, note, harmony, tempo
import re
import sqlite3
import pandas as pd


def music_xml_to_data(xml_folder):

    """
    Fonction pour récupérer notes, onset, duration, chord

    Args:
        xml_folder(string): path to the xml folder 
    Returns:
        list(list)): [[]]
    """
    all_notes = []
    for filename in os.listdir(xml_folder):
        if not filename.lower().endswith('.xml'):
            continue
        
        path = os.path.join(xml_folder, filename)
        score = converter.parse(path)
        # Extraire les accords
        chord_symbols = sorted(
            score.flat.getElementsByClass(harmony.ChordSymbol),
            key=lambda cs: cs.offset
        )
        
        mm = score.flat.getElementsByClass(tempo.MetronomeMark)
        if mm:
            bpm = mm[0].number
        else:
            bpm = 120  # valeur par défaut si pas d’indication

        sec_per_quarter = 60.0 / bpm

        chord_data = [(cs.getOffsetInHierarchy(score) * sec_per_quarter, cs.figure) 
                      for cs in chord_symbols]
        chord_data.sort(key=lambda x: x[0])
        
        for n in score.recurse().getElementsByClass(note.Note):
            offset = float(n.getOffsetInHierarchy(score)) * sec_per_quarter
            dur = float(n.duration.quarterLength) * sec_per_quarter
            midi_pitch = n.pitch.midi
            
            # Attribution d'accord par recherche de plage temporelle
            chord_symbol = None
            for i, (chord_offset, chord_fig) in enumerate(chord_data):
                if chord_offset <= offset:
                    chord_symbol = chord_fig
                else:
                    break
            
            all_notes.append([midi_pitch, offset, dur, chord_symbol])
    
    return all_notes


def extract_for_melid(conn, melid):
    notes = pd.read_sql_query(
        "SELECT onset, pitch, duration FROM melody WHERE melid = ? ORDER BY onset",
        conn, params=(melid,)
    )
    beats = pd.read_sql_query(
        "SELECT onset AS beat_onset, chord AS chord_label FROM beats WHERE melid = ? ORDER BY beat_onset",
        conn, params=(melid,)
    )
    merged = pd.merge_asof(
        notes, beats,
        left_on='onset', right_on='beat_onset',
        direction='backward'
    )

    # Remplacer les chaînes vides et 'NC' par NaN pour pouvoir les traiter
    merged['chord_label'] = merged['chord_label'].replace(['', 'NC'], pd.NA)
    
    # Propagation vers l'avant (forward fill) pour remplir les accords manquants
    merged['chord_label'] = merged['chord_label'].ffill()
    
    # Si il reste encore des NaN au début (notes avant le premier accord), on met 'NA'
    merged['chord_label'] = merged['chord_label'].fillna('NA')
    
    # Supprimer les lignes où 'chord_label' est 'NA'
    merged = merged[merged['chord_label'] != 'NA']
    
    # Retourner une liste de listes [pitch, onset, duration, accord]
    return merged[['pitch','onset','duration','chord_label']].values.tolist()


def extract_all_flat(db_path):
    conn = sqlite3.connect(db_path)
    melids = pd.read_sql_query("SELECT DISTINCT melid FROM melody", conn)['melid'].tolist()

    all_notes = []
    for mid in melids:
        notes = extract_for_melid(conn, mid)
        all_notes.extend(notes)  # ajoute de nouvelles notes à la liste principale

    conn.close()
    return all_notes

# Mapping from note names to semitone indices (C=0, C#/Db=1, ..., B=11)
NOTE_TO_SEMITONE = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'E#': 5, 'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11
}
SEMITONE_TO_NOTE = {v: k for k, v in NOTE_TO_SEMITONE.items() if len(k) == 1 or '#' in k}


def split_chord(chord_label):
    """
    Sépare une étiquette d'accord en racines et type.
    Gère accords simples (e.g. 'C7') et slash-chords (e.g. 'C/Dm7').
    Retourne:
      roots: liste de racines (['C'] ou ['C','D'])
      chord_type: type d'accord (e.g. '7', 'm7', '')
    """
    if '/' in chord_label:
        parts = chord_label.split('/')
        roots = []
        chord_type = None
        for p in parts:
            m = re.match(r'^([A-G][#b]?)(.*)$', p)
            if not m:
                raise ValueError(f"Accord invalide: {p}")
            root, ctype = m.group(1), m.group(2)
            roots.append(root)
            if chord_type is None:
                chord_type = ctype
        return roots, chord_type or ''
    else:
        m = re.match(r'^([A-G][#b]?)(.*)$', chord_label)
        if not m:
            raise ValueError(f"Accord invalide: {chord_label}")
        return [m.group(1)], m.group(2)


def transpose_note(note, semitones):
    """
    Transpose une note donnée de semitones.
    """
    if note not in NOTE_TO_SEMITONE:
        raise ValueError(f"Note inconnue: {note}")
    idx = (NOTE_TO_SEMITONE[note] + semitones) % 12
    return SEMITONE_TO_NOTE[idx]


def transpose_chord(chord_label, semitones):
    """
    Transpose l'étiquette d'accord complète.
    """
    roots, ctype = split_chord(chord_label)
    transposed = [transpose_note(r, semitones) for r in roots]
    if len(transposed) == 2:
        return f"{transposed[0]}{ctype}/{transposed[1]}{ctype}"
    return f"{transposed[0]}{ctype}"


def augment_dataset(dataset, max_semitones=6):
    """
    Génère des jeux de données transposés de -max_semitones à +max_semitones (sauf 0).
    """
    augmented = []
    for shift in range(-max_semitones, max_semitones+1):
        if shift == 0:
            continue
        for note in dataset:
            pitch, onset, duration, chord = note
            new_pitch = pitch + shift
            new_chord = transpose_chord(chord, shift)
            augmented.append([new_pitch, onset, duration, new_chord])
    return augmented

mapping = {
    '': 'maj',    # fondamentale seule → majeur
    '+j7': 'maj7',   # j7 → maj7
    'j7': 'maj7',
    '-': 'min',    # tiret seul → mineur
    '-m': 'min',
    '-dim': 'dim',    # -dim → dim
    'o': 'dim',    # cercle seul → dim
    'o7': 'dim7',   # o7 → dim7
    '+': 'aug',    # + → aug
    '-j7': 'minmaj7',# -j7 → minmaj7
    'ø7': 'm7b5',   # demi‑diminué → m7b5
}

def normalize_chord(chord_label):
    # 1. Extraire la fondamentale (root)
    m = re.match(r'^([A-G](?:#|b)?)', chord_label)
    root = m.group(1) if m else ''
    
    # 2. Isoler le type (tout ce qui suit la root, avant un éventuel '/')
    rest = chord_label[len(root):]
    typ = rest.split('/')[0]  # on coupe avant le slash
    
    # 3. Appliquer le mapping réduit si nécessaire
    norm_type = mapping.get(typ, typ)
    
    # 4. Reconstruire :
    #    - si 'maj', on considère root seul (accord majeur)
    #    - sinon, root + type
    if norm_type == 'maj' or norm_type == '':
        return root
    else:
        return root + norm_type

# Exemple d'application sur dataset_final
def uniformize_dataset(dataset_final):
    normalized = []
    for pitch, onset, duration, chord in dataset_final:
        new_chord = normalize_chord(chord)
        normalized.append([pitch, onset, duration, new_chord])
    return normalized

def parse_chord_label(label):
    """Sépare un label comme 'C#-7911' en ('C#', '-7911')."""
    # Racine = un ou deux caractères (A-G suivi optionnel # ou b)
    m = re.match(r'([A-G][#b]?)(.*)', label)
    if m:
        root, quality = m.group(1), m.group(2)
    else:
        root, quality = label, ''
    # Normalisation des cas enharmoniques
    enharmonic_map = {
        'B#': 'C', 'E#': 'F', 'Cb': 'B', 'Fb': 'E'
    }
    root = enharmonic_map.get(root, root)
    # Si quality vide, on représente majeur par ''
    return root, quality

def build_chord_sequences(raw_data, seq_length=64):
    """
    Transforme la liste raw_data en listes de séquences d'accords (root,type).
    raw_data: liste de listes [pitch, onset, dur, chord]
    seq_length: longueur fixe souhaitée pour découpage
    Retourne: list of sequences, each sequence is list of (root,type)
    """
    # Extraire juste les labels d'accords en ordre
    chords = [entry[3] for entry in raw_data]
    # Parse labels
    parsed = [parse_chord_label(lbl) for lbl in chords]
    # Découper en sous-séquences de taille seq_length
    sequences = []
    for i in range(0, len(parsed), seq_length):
        sequences.append(parsed[i:i+seq_length])
    print("==============sequence_built==============")
    return sequences

