from music21 import * # type:ignore
from typing import List, Dict, Optional, Set
from time import sleep
from markov import build_vlmc_table, generate_symbol_vlmc, symbol_to_key, truncate_key
import os
from chord_extractor.extractors import Chordino
from chord_extractor.extractors import TuningMode 
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import tempfile
import re

def preprocess_audio(y, sr):
    """Preprocess audio to improve chord detection"""
    
    # 1. Apply high-pass filter to remove low-frequency noise/drums
    nyquist = sr / 2
    low_cutoff = 80 / nyquist  # Remove frequencies below 80Hz
    b, a = butter(4, low_cutoff, btype='high')  #type:ignore
    y_filtered = filtfilt(b, a, y)
    
    # 2. Gentle compression to even out dynamics
    y_compressed = np.sign(y_filtered) * np.power(np.abs(y_filtered), 0.7)
    
    # 3. Normalize
    y_normalized = y_compressed / np.max(np.abs(y_compressed))
    
    return y_normalized

def post_process_chords(labels, min_duration=0.5):
    """Clean up chord sequence by removing very short chords"""
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

        # First try with adaptive tuning - MUCH more sensitive parameters
    chordino_adaptive = Chordino(
        use_nnls=True,
        roll_on=1,  # Reduced from 5 - less temporal smoothing
        spectral_whitening=0.3,  # Reduced from 0.9 - keep more harmonics
        spectral_shape=0.9,  # Increased from 0.5 - less aggressive shaping
        boost_n_likelihood=0.5,  # Reduced from 1.5 - much less conservative
        tuning_mode=TuningMode.LOCAL
    )

    # Then try with global tuning - same sensitive parameters
    chordino_global = Chordino(
        use_nnls=True,
        roll_on=1,
        spectral_whitening=0.3,
        spectral_shape=0.9,
        boost_n_likelihood=0.5,
        tuning_mode=TuningMode.GLOBAL
    )

    # Also try default parameters for comparison
    chordino_default = Chordino()

    labels_adaptive = chordino_adaptive.extract(temp_path)
    labels_global = chordino_global.extract(temp_path)
    labels_default = chordino_default.extract(temp_path)

    try:
        os.remove(temp_path)
    except OSError:
        pass

    # Count non-N chords for each
    non_n_adaptive = sum(1 for label in labels_adaptive if label.chord != 'N')
    non_n_global = sum(1 for label in labels_global if label.chord != 'N')
    non_n_default = sum(1 for label in labels_default if label.chord != 'N')

    # Choose the result with most actual chord detections
    if non_n_adaptive >= non_n_global and non_n_adaptive >= non_n_default:
        labels = labels_adaptive
        print("Using adaptive tuning results")
    elif non_n_global >= non_n_default:
        labels = labels_global
        print("Using global tuning results")
    else:
        labels = labels_default
        print("Using default parameters results")

    # Post-process to remove very short chord detections
    labels_filtered = post_process_chords(labels, min_duration=0.3)
    chord_list = [
    (ch.chord, ch.timestamp)
    for ch in labels_filtered
    if ch.chord != 'N'
    ]
    return chord_list


def get_pitches_by_chord(
        folder: str,
        chords):
    """
    Prends une grille, créé un dictionnaire avec l'accord et les notes

    Args:
        folder: le chemin du dossier dans lequel sont contenues les fichiers musicxml
        chords: la grille
    Returns:
        chord_map: un dictionnaire contenant les accords uniques avec la liste de notes
    """
    if chords and isinstance(chords[0],tuple):
        chords = [C[0] for C in chords]
    
    chord_map: Dict[str, List[int]] = {chord: [] for chord in set(chords)}

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
            # Si c'est un accord qu'on suit, on stocke la hauteur MIDI
            if cs and cs.figure in chord_map:
                chord_map[cs.figure].append(element.pitch.midi)

    return chord_map

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