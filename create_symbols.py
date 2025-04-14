import mido
import os
import json
from tqdm import tqdm
import time
import polars as pl

"""
Ce fichier contient deux fonctions :
- Extract features qui récupère les messages d'un fichiers midi
- Find_midi_files qui parse un dataset (Maestro)
"""
def extract_features(midi_file, output='nested_list'):
    """
    Extrait les paramètres de note d'un fichier MIDI et renvoie une matrice.
    
    Args:
        midi_file: Chemin vers le fichier MIDI.
        output: Format de sortie, 'nested_list' ou 'polars'.
        
    Returns:
        Une matrice contenant les notes avec colonnes [pitch, onset, duration].
    """
    try:
        # Charger le fichier MIDI
        mid = mido.MidiFile(midi_file)
        notes = []
        active_notes = {}  # clé: (track, channel, note) -> valeur: {'start_time': ..., 'velocity': ...}

        # Parcourir toutes les pistes du fichier MIDI
        for track_index, track in enumerate(mid.tracks):
            current_time = 0
            for msg in track:
                current_time += msg.time

                # Ignorer les messages liés au tempo ou à la signature temporelle
                if msg.type in ['set_tempo', 'time_signature']:
                    continue

                # Note-on (activation)
                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_index, msg.channel, msg.note)
                    active_notes[key] = {
                        'start_time': current_time,
                        'velocity': msg.velocity
                    }
                # Note-off (ou note_on avec vélocité nulle)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (track_index, msg.channel, msg.note)
                    if key in active_notes:
                        start_info = active_notes.pop(key)
                        start_time = start_info['start_time']
                        duration_ticks = current_time - start_time
                        # On ne conserve que pitch, onset et duration
                        notes.append({
                            'pitch': msg.note,
                            'onset': start_time,
                            'duration': duration_ticks,
                            'velocity': start_info['velocity']
                        })

        # Trier les notes par onset (temps de début)
        notes = sorted(notes, key=lambda x: x['onset'])
        # Construire la matrice sous forme de liste de listes
        mnotes = [[n['pitch'], n['onset'], n['duration'], n['velocity']] for n in notes]

        if output == 'nested_list':
            return mnotes
        elif output == 'polars':
            return pl.DataFrame(mnotes, schema=['pitch', 'onset', 'duration', 'velocity'], orient="row")
        else:
            raise ValueError("Le paramètre output doit être 'nested_list' ou 'polars'")
    except Exception as e:
        print("Erreur lors de l'extraction:", e)
        return None

def find_midi_files(base_dir):
    """
    Recherche récursivement tous les fichiers MIDI dans un répertoire.
    Args:
        base_dir: Répertoire de base à parcourir
    Returns:
        Liste des chemins de fichiers MIDI trouvés
    """
    midi_files = []
    print(f"Recherche de fichiers MIDI dans {base_dir}...")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                midi_path = os.path.join(root, file)
                midi_files.append(midi_path)
    print(f"Total de {len(midi_files)} fichiers MIDI trouvés.")
    return midi_files

def create_symbole(matrix):
    """
    On prend le dico en entrée puis on en ressort un tuple par notes
    """
    if isinstance(matrix,pl.DataFrame):
        return matrix.select(["pitch", "duration", "velocity"]).rows()
    else:
        return [(note[0], note[2], note[3]) for note in matrix]
