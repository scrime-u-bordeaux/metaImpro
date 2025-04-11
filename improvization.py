import os
import json
import numpy as np
import random
from typing import List, Dict, Optional, Any
import mido
import time
from factor_oracle import FactorOracle  

def extract_features(midi_file: str) -> Optional[Dict[str, Any]]:
    """
    Extracts features from a MIDI file and returns flattened symbols.
    Chords are detected by grouping notes whose start times differ by less than a fixed threshold (in ticks).
    """
    try:
        mid = mido.MidiFile(midi_file)
        notes = []
        active_notes = {}  # key: (track, channel, note)
        
        for track_index, track in enumerate(mid.tracks):
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type in ['set_tempo', 'time_signature']:
                    continue
                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_index, msg.channel, msg.note)
                    active_notes[key] = {'start_time': current_time, 'velocity': msg.velocity}
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (track_index, msg.channel, msg.note)
                    if key in active_notes:
                        start_info = active_notes.pop(key)
                        start_time = start_info["start_time"]
                        duration_ticks = current_time - start_time
                        notes.append({
                            'pitch': msg.note,
                            'start_ticks': start_time,
                            'duration_ticks': duration_ticks,
                            'velocity': start_info['velocity']
                        })
        notes = sorted(notes, key=lambda x: x['start_ticks'])
        
        # Group notes into chords using a threshold (in ticks)
        CHORD_THRESHOLD_TICKS = 10  # increased threshold to better detect chords
        chords = []
        current_chord = []
        for note in notes:
            if not current_chord:
                current_chord.append(note)
            elif abs(note['start_ticks'] - current_chord[0]['start_ticks']) <= CHORD_THRESHOLD_TICKS:
                current_chord.append(note)
            else:
                chords.append(current_chord)
                current_chord = [note]
        if current_chord:
            chords.append(current_chord)
        
        # Label each note with its chord_id and mark if it's part of a chord
        for i, chord in enumerate(chords):
            for note in chord:
                note['chord_id'] = i
                note['in_chord'] = (len(chord) > 1)
        
        symbols = []
        for i, note in enumerate(notes):
            symbol = {
                'id': i,
                'pitch': note['pitch'],
                'duration': note['duration_ticks'],
                'velocity': note['velocity'],
                'in_chord': note['in_chord'],
                'chord_id': note['chord_id']
            }
            if note['duration_ticks'] < 100:
                symbol['duration_category'] = 'très court'
            elif note['duration_ticks'] < 300:
                symbol['duration_category'] = 'court'
            elif note['duration_ticks'] < 600:
                symbol['duration_category'] = 'moyen'
            elif note['duration_ticks'] < 1000:
                symbol['duration_category'] = 'long'
            else:
                symbol['duration_category'] = 'très long'
            symbols.append(symbol)
        
        result = {
            'file_name': os.path.basename(midi_file),
            'full_path': midi_file,
            'ticks_per_beat': mid.ticks_per_beat,
            'num_notes': len(notes),
            'num_chords': len(chords),
            'symbols': symbols
        }
        return result
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {midi_file}: {str(e)}")
        return None

def process_maestro_file(midi_file: str, oracle: FactorOracle) -> List[Dict[str, Any]]:
    """
    Processes a Maestro MIDI file to extract flattened symbols.
    """
    features = extract_features(midi_file)
    if features is None:
        return []
    return features.get("symbols", [])

class FOGenerator:
    """
    Factor Oracle-based improvisation generator using a heuristic inspired by 'Navigating the Oracle'.

    Enhancements:
    - Uses suffix links instead of factor links for smoother transitions.
    - Employs reverse suffix links for better navigation and diversity.
    - Chooses transitions based on context length, optimizing musical coherence.
    - Implements a continuity factor to balance novelty and structure.
    - Uses a taboo list to prevent excessive looping and repetition.
    """

    def __init__(self, oracle: FactorOracle, continuity_factor=10, taboo_length=5):
        self.oracle = oracle
        self.current_state = 0
        self.chord_mapping: Dict[Any, List[Dict[str, Any]]] = {}
        self.continuity_factor = continuity_factor
        self.taboo_list = []
        self.taboo_length = taboo_length  # Number of steps before revisiting a state

    def generate_next(self, last_input_symbol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates the next symbol based on enhanced heuristic navigation."""
        if last_input_symbol is not None:
            symbol_hash = self.oracle._get_symbol_hash(last_input_symbol)
            for state, transitions in enumerate(self.oracle.transitions):
                if symbol_hash in transitions:
                    self.current_state = transitions[symbol_hash]
                    break

        # Candidate 1: Sequential reading (i+1)
        seq_candidate = self.current_state + 1 if self.current_state + 1 < len(self.oracle.symbols) else None

        # Candidate 2: Suffix jump (if a suffix link exists)
        suffix_candidate = None
        if self.oracle.suffix_links[self.current_state] is not None:
            candidate = self.oracle.suffix_links[self.current_state]
            if candidate + 1 < len(self.oracle.symbols):
                suffix_candidate = candidate + 1

        # Reverse suffix links exploration (for diversity)
        reverse_suffix_candidates = [
            i for i, link in enumerate(self.oracle.suffix_links) if link == self.current_state
        ]
        reverse_suffix_candidate = max(reverse_suffix_candidates, key=lambda x: self.oracle.lrs[x], default=None)

        # Choose the best candidate based on context length
        best_candidate = None
        best_lrs = -1

        for candidate in [seq_candidate, suffix_candidate, reverse_suffix_candidate]:
            if candidate is not None and candidate not in self.taboo_list:
                lrs_value = self.oracle.lrs[candidate]
                if lrs_value > best_lrs:
                    best_candidate = candidate
                    best_lrs = lrs_value

        if best_candidate is None:
            best_candidate = self.current_state  # Fallback to current state if no valid move is found

        # Update state and taboo list
        self.current_state = best_candidate
        self.taboo_list.append(self.current_state)
        if len(self.taboo_list) > self.taboo_length:
            self.taboo_list.pop(0)

        # Retrieve symbol and ensure chord coherence
        original = self.oracle.symbols[self.current_state]
        symbol = {
            'pitch': original.get('pitch', 60),
            'duration': original.get('duration', 480),
            'velocity': original.get('velocity', 64),
            'duration_category': original.get('duration_category', 'inconnu'),
            'in_chord': original.get('in_chord', False),
            'chord_id': original.get('chord_id', None)
        }

        if symbol['in_chord'] and symbol['chord_id'] is not None:
            chord_group = self.chord_mapping.get(symbol['chord_id'], [])
            chord_pitches = [s['pitch'] for s in chord_group if s.get('pitch') is not None]
            if len(chord_pitches) > 1:
                symbol['chord'] = chord_pitches

        return symbol

    def generate_sequence(self, seed_symbols: List[Dict[str, Any]], length: int) -> List[Dict[str, Any]]:
        """Generates a sequence starting from the seed symbols extracted from a MIDI file."""
        self.chord_mapping = {}
        for sym in seed_symbols:
            if sym.get('in_chord') and sym.get('chord_id') is not None:
                cid = sym['chord_id']
                if cid not in self.chord_mapping:
                    self.chord_mapping[cid] = []
                self.chord_mapping[cid].append(sym)

        self.current_state = 0
        generated_sequence = seed_symbols.copy()

        # Process the seed symbols to establish initial context
        for symbol in seed_symbols:
            self.generate_next(symbol)

        # Generate new symbols based on navigation heuristic
        for _ in range(length):
            next_symbol = self.generate_next()
            generated_sequence.append(next_symbol)

        return generated_sequence

def play_generated_sequence(sequence: List[Dict[str, Any]], output_midi: str = 'generated_sequence.mid', inter_onset: int = 100) -> None:
    """
    Converts a sequence of symbols into a MIDI file.

    - Uses absolute timing to preserve chord overlaps.
    - Generates note-on and note-off events based on duration.
    - Outputs the generated sequence as a MIDI file.
    """
    events = []  # List of (absolute_time, message)

    for i, symbol in enumerate(sequence):
        onset = i * inter_onset  # Absolute onset time for the note

        # Schedule note_on for the main note
        events.append((onset, mido.Message('note_on', note=symbol['pitch'], velocity=symbol['velocity'])))

        # If it's a chord, schedule all its note_on events
        if 'chord' in symbol:
            for pitch in symbol['chord']:
                events.append((onset, mido.Message('note_on', note=pitch, velocity=symbol['velocity'])))

        # Schedule note_off events after duration
        off_time = onset + int(symbol['duration'])
        events.append((off_time, mido.Message('note_off', note=symbol['pitch'], velocity=0)))

        # Ensure all chord notes get their note_off events
        if 'chord' in symbol:
            for pitch in symbol['chord']:
                events.append((off_time, mido.Message('note_off', note=pitch, velocity=0)))

    # Sort events by absolute time
    events.sort(key=lambda x: x[0])

    # Convert absolute times to relative times and add events to track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message('program_change', program=0, time=0))

    last_time = 0
    for abs_time, msg in events:
        msg.time = abs_time - last_time  # Convert to relative timing
        last_time = abs_time
        track.append(msg)

    # Save the generated MIDI
    mid.save(output_midi)
    print(f"Generated MIDI saved at {output_midi}")

    """
    Plays and saves a generated sequence by creating a MIDI file.
    This version uses absolute scheduling for note events so that overlapping notes (and chords) occur.
    
    Each symbol's onset is set to index * inter_onset, and its note_off is scheduled at (onset + duration).
    Then, events are sorted by absolute time and converted to relative times.
    """
    events = []  # list of (absolute_time, message)
    for i, symbol in enumerate(sequence):
        onset = i * inter_onset
        # Schedule note_on for the main note
        events.append((onset, mido.Message('note_on', note=symbol['pitch'], velocity=symbol['velocity'])))
        # If symbol contains a chord field, schedule note_on for all chord notes
        if 'chord' in symbol:
            for pitch in symbol['chord']:
                events.append((onset, mido.Message('note_on', note=pitch, velocity=symbol['velocity'])))
        # Schedule note_off events at onset + duration
        off_time = onset + int(symbol['duration'])
        events.append((off_time, mido.Message('note_off', note=symbol['pitch'], velocity=0)))
        if 'chord' in symbol:
            for pitch in symbol['chord']:
                events.append((off_time, mido.Message('note_off', note=pitch, velocity=0)))
    
    # Sort events by absolute time
    events.sort(key=lambda x: x[0])
    
    # Convert absolute times to relative times
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # Add a program change at time=0
    track.append(mido.Message('program_change', program=0, time=0))
    last_time = 0
    for abs_time, msg in events:
        msg.time = abs_time - last_time
        last_time = abs_time
        track.append(msg)
    
    mid.save(output_midi)
    print(f"Séquence sauvegardée dans {output_midi}")
