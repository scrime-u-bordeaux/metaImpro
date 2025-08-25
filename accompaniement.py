from music21 import * # type:ignore
from typing import List, Dict, Optional, Set
from time import sleep
from markov import build_vlmc_table, generate_symbol_vlmc, symbol_to_key, truncate_key
import os
import re


"""
def preprocess_audio(y, sr):
    #Preprocess audio to improve chord detection#
    
    # 1. Apply high-pass filter to remove low-frequency noise/drums
    nyquist = sr / 2
    low_cutoff = 80 / nyquist  # Remove frequencies below 80Hz
    b, a = butter(4, low_cutoff, btype='high')  #type:ignore
    y_filtered = filtfilt(b, a, y)
    
    # 2. Gentle compression to even out dynamics
    y_compressed = np.sign(y_filtered) * np.power(np.abs(y_filtered), 0.6)
    
    # 3. Normalize
    y_normalized = y_compressed / np.max(np.abs(y_compressed))
    
    return y_normalized

def post_process_chords(labels, min_duration=0.5):
    #Clean up chord sequence by removing very short chords#
    if not labels:
        return labels
    
    filtered_labels = []
    current_chord = labels[0]
    
    for i in range(1, len(labels)):
        duration = labels[i].timestamp - current_chord.timestamp
        
        if duration >= min_duration:
            filtered_labels.append(current_chord)
            current_chord = labels[i]
        else:
            # Skip this chord, it's too short
            continue
    
    # Don't forget the last chord
    filtered_labels.append(current_chord)
    
    return filtered_labels

def get_chord_progression(wav_file):
    a = time.time()
    # Load and preprocess audio
    y, sr = librosa.load(wav_file, sr=None)

    # Trim the beginning (drums)
    start_trim = int(0* sr)
    y_trimmed = y[start_trim:]

    # Preprocess audio
    y_processed = preprocess_audio(y_trimmed, sr)

    # Create tempfile audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
    sf.write(temp_path, y_processed, sr)

    chordino_default = Chordino()

    labels= chordino_default.extract(temp_path)

    try:
        os.remove(temp_path)
    except OSError:
        pass
    # Post-process to remove very short chord detections
    labels_filtered = post_process_chords(labels, min_duration=1)
    chord_list = [
    (ch.chord, ch.timestamp)
    for ch in labels_filtered
    if ch.chord != 'N'
    ]
    b = time.time()
    print("time = ",b-a)
    return chord_list
    
"""

def split_chord_figure(chord):

    CHORD_ROOT_RE = re.compile(r'^([A-Ga-g][#b]?)(.*)$')
    figure = chord.split('/')[0].strip()
    m = CHORD_ROOT_RE.match(figure)
    root = m.group(1)
    type = m.group(2).strip()
    return root, type


def get_pitches_by_chord(
    folder: str,
    chords,
    group_by_type=False,
    max_notes_per_chord=None):
    """
    Prends une grille, créé un dictionnaire avec l'accord et les notes
    Args:
        folder: le chemin du dossier dans lequel sont contenues les fichiers musicxml
        chords: la grille
        group_by_type: if True, group by chord type instead of full chord name
        max_notes_per_chord: maximum number of notes to collect per chord (None for no limit)
    Returns:
        chord_map: un dictionnaire contenant les accords uniques avec la liste de notes
    """
    if chords and isinstance(chords[0], tuple):
        chords = [C[0] for C in chords]
    
    # Build keys depending on mode
    if group_by_type:
        # collect types present in the progression
        types = set()
        for ch in (chords or []):
            try:
                root, chord_type = split_chord_figure(ch)  # Fixed: proper unpacking
                types.add(chord_type)  # Add the chord type to the set
            except Exception:
                continue
        # remove empty type keys if you prefer, but keep them for completeness
        chord_map: Dict[str, List[int]] = {t: [] for t in types}
    else:
        chord_map: Dict[str, List[int]] = {chord: [] for chord in set(chords or [])}
    
    target_C = pitch.Pitch('C')  # target for normalization
    
    # 1) Load the XML and extract pitch lists
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.xml')):
            continue
        path = os.path.join(folder, filename)
        score = converter.parse(path)
        
        # Pour chaque note (ou rest) dans la partition…
        for element in score.recurse().notesAndRests:
            # Ne traiter que les vraies notes
            if not isinstance(element, note.Note):
                continue
            
            # Récupère son ChordSymbol de contexte (s'il y en a un)
            cs = element.getContextByClass(harmony.ChordSymbol)
            
            if not cs:  # Skip if no chord symbol
                continue
                
            # Si c'est un accord qu'on suit, on stocke la hauteur MIDI
            cs_root, cs_type = split_chord_figure(cs.figure)  # Fixed: consistent naming
            
            if group_by_type:
                if cs_type in chord_map:
                    # Skip if we've already collected enough notes for this chord type
                    if max_notes_per_chord and len(chord_map[cs_type]) >= max_notes_per_chord:
                        continue
                        
                    # compute transposition in semitones so that cs_root -> C
                    root_pitch = pitch.Pitch(cs_root)
                    # Use pitchClass difference (works across octaves)
                    transposition_semitones = (target_C.midi % 12) - (root_pitch.midi % 12)
                    normalized_midi = element.pitch.midi + transposition_semitones
                    chord_map[cs_type].append(int(normalized_midi))
            else:
                if cs.figure in chord_map:
                    # Skip if we've already collected enough notes for this chord
                    if max_notes_per_chord and len(chord_map[cs.figure]) >= max_notes_per_chord:
                        continue
                        
                    chord_map[cs.figure].append(element.pitch.midi)
    
    return chord_map

def note_to_semitone(note):
    """Convertit une note (C, D, E, etc.) en nombre de demi-tons depuis C"""
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    return note_map.get(note, 0)

def transpose_chord_data(chord_data, progression):
    """
    Transpose les données d'accords selon la progression donnée
    
    Args:
        chord_data: dictionnaire avec les types d'accords et leurs notes MIDI
        progression: liste d'accords de la progression
    
    Returns:
        dictionnaire avec les accords transposés
    """
    result = {}
    
    # Obtenir tous les accords uniques de la progression
    unique_chords = list(set(progression))
    
    for chord in unique_chords:
        # Extraire la note fondamentale et le type d'accord
        root_note = chord[0]  # Première lettre (C, F, G, etc.)
        chord_type = chord[1:]  # Le reste (7, maj7, 6, etc.)
        
        # Calculer l'intervalle de transposition
        semitones = note_to_semitone(root_note)
        
        # Si le type d'accord existe dans les données de base
        if chord_type in chord_data:
            # Transposer toutes les notes
            transposed_notes = [note + semitones for note in chord_data[chord_type]]
            result[chord] = transposed_notes
        else:
            # Si le type d'accord n'existe pas, créer une liste vide
            result[chord] = []
    
    return result


def chord_loop(synth,
               stop_event,
               progression: List[str],
               bpm: int = 120,
               velocity: int = 80,
               lower_octave: int = 12,
               pattern = None,
               riff_pattern = None,
               log_callback=None):
    """
    Play an endless piano-stride 4/4 loop over C7 and C#7.

    On each bar:
      - Beats 1 & 3: left-hand bass (root one octave down)
      - Beats 2 & 4: left-hand chord (root position)

    Args:
      synth:      fluidsynth.Synth already initialized.
      stop_event: threading.Event to break the loop.
      progression:    sequence d'accords, ex ["C7","F7","G7",…].
      bpm:        beats per minute.
      velocity:   MIDI velocity for all notes.
      lower_octave: number of semitones to transpose left-hand bass down.
      log_callback: optional fn(str) to receive log messages.
    """
    if pattern is None:
        pattern = ["bass", "chord", "fifth", "chord"]

    riff_subdivision = 2
    # rythme de base et subdivision pour riff
    beat_duration = 60.0 / bpm
    sub_dur = beat_duration / riff_subdivision if riff_pattern else None
    bar_index = 0

    while not stop_event.is_set():
        chord_name = progression[bar_index % len(progression)]
        # extrait fundamental et force octave 4
        root_symbol = chord_name.rstrip('0123456789')
        root_sym = f"{root_symbol}4"
        try:
            root_midi = pitch.Pitch(root_sym).midi
        except Exception:
            if log_callback:
                log_callback(f"Impossible de parser '{root_sym}'")
            bar_index += 1
            continue

        # calcul des intervalles de l'accord
        third = root_midi + 4
        fifth = root_midi + 7
        seventh = root_midi + 10
        voicing = [root_midi, third, fifth, seventh]

        if riff_pattern:
            # joue le riff en boucle sur la mesure
            for interval in riff_pattern:
                if stop_event.is_set(): break
                note = root_midi + interval
                synth.noteon(0, note, velocity)
                if log_callback:
                    log_callback(f"Riff on {chord_name}: interval {interval}, note {note}")
                sleep(sub_dur) #type:ignore
                synth.noteoff(0, note)
        else:
            # joue le pattern stride en noires
            for beat_idx, part in enumerate(pattern, start=1):
                if stop_event.is_set(): break
                if part == "bass":
                    notes = [root_midi - lower_octave]
                elif part == "fifth":
                    notes = [fifth - lower_octave]
                else:
                    notes = voicing
                for note in notes:
                    synth.noteon(0, note, velocity)
                if log_callback:
                    log_callback(f"Bar {bar_index+1}, beat {beat_idx}, {chord_name}/{part}: {notes}")
                sleep(beat_duration)
                for note in notes:
                    synth.noteoff(0, note)

        bar_index += 1


def make_vlmc_for_chord(symbol_sequences, max_order=3, similarity_level=1):
    """
    Given a dict mapping chord name → list of symbols (notes/chords) from your corpus,
    build a VLMC table and collect all possible keys.
    Returns a dict: chord_name → (vlmc_table, all_keys).
    """
    vlmcs = {}
    
    for chord_name, seq in symbol_sequences.items():
        # 1) build the VLMC table over the raw sequence
        table = build_vlmc_table(seq,
                                 max_order=max_order,
                                 similarity_level=similarity_level)

        # 2) build your fallback key‑list from the raw symbols
        #    (so even symbols that never occur as "successors" are included)
        keyset = {
            truncate_key(symbol_to_key(sym), similarity_level)
            for sym in seq
        }
        all_keys = list(keyset)

        vlmcs[chord_name] = (table, all_keys)

    return vlmcs