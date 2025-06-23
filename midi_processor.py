import pretty_midi
import os
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import copy

"""
Ce fichier sert  à traiter les morceaux midi pour les transformer en symbols :
On retrouve une fonction pour parse un dataset et des fichiers.
Ensuite, il y a une fonction qui permet créer un dataset des features midi.
Puis, une fonction pour transformer ce dataset en symboles utilisables pour la suite du projet

Midi Toolkit : 
Agrégation des notes proches en accords, filtre legato.
"""

class MidiSymbolProcessor:
    """
    Process MIDI files to extract musical symbols (notes and chords)
    with an efficient representation for real-time generation.
    """
    
    def __init__(self, inner_chord_threshold_ms: float = 50.0, 
                maximum_chord_threshold_ms: float = 200.0,
                include_chords = True):
        """
        Initialize the MIDI symbol processor.
        
        Args:
            inner_chord_threshold_ms: Maximum time difference (ms) between notes to be considered part of the same chord
            maximum_chord_threshold_ms: Maximum time difference (ms) between the first and the last note to be considered part of the same chord
        """
        self.inner_chord_threshold_ms = inner_chord_threshold_ms
        self.maximum_chord_threshold_ms = maximum_chord_threshold_ms
    

    def find_midi_files(self, base_dir: str) -> List[str]:
        """
        Recursively find all MIDI files in a directory.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            List of MIDI file paths
        """
        midi_files = []
        print(f"Searching for MIDI files in {base_dir}...")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.mid', '.midi')):
                    midi_path = os.path.join(root, file)
                    midi_files.append(midi_path)
        print(f"Found {len(midi_files)} MIDI files.")
        return midi_files
    
    def extract_notes(self, midi_file:str) -> List[Dict]:
        """
        Extract note features from a MIDI file using pretty_midi.
        
        Args:
            midi_file: Path to the MIDI file
            
        Returns:
            List of dictionaries containing note data
        """
        try:
            # Load the MIDI file with pretty_midi
            pm = pretty_midi.PrettyMIDI(midi_file)
            
            # Extract all notes from all instruments
            notes = []
            for instrument in pm.instruments:
                # Skip drum tracks if needed
                if instrument.is_drum:
                    continue
                    
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'onset': note.start * 1000,  # Convert to ms
                        'duration': (note.end - note.start) * 1000,  # Convert to ms
                        'velocity': note.velocity,
                        'instrument': instrument.program
                    })
            
            # Sort notes by onset time
            notes.sort(key=lambda x: x['onset'])
            return notes
            
        except Exception as e:
            print(f"Error extracting notes from {midi_file}: {e}")
            return []
        
    def create_symbols(self, notes: List[Dict]) -> List[Dict]:
        """
        Create symbol representations (notes and chords) from extracted note data.
        
        Args:
            notes: List of note dictionaries from extract_notes
            
        Returns:
            List of symbol dictionaries (notes or chords)
        """
        if not notes:
            return []
            
        symbols = []
        # Group notes by onset time (with threshold)
        onset_groups = defaultdict(list)
        
        for note in notes:
            # Find the group this note belongs to
            assigned = False
            for onset_time in list(onset_groups.keys()):
                # Check if this note is within the inner threshold of the group's reference time
                if abs(note['onset'] - onset_time) <= self.inner_chord_threshold_ms:
                    # Check if adding this note would exceed the maximum chord time span
                    group_notes = onset_groups[onset_time]
                    earliest_onset = min([n['onset'] for n in group_notes] + [onset_time])
                    latest_onset = max([n['onset'] for n in group_notes] + [note['onset']])
                    
                    # Only add to group if total time span is within maximum threshold
                    if latest_onset - earliest_onset <= self.maximum_chord_threshold_ms:
                        onset_groups[onset_time].append(note)
                        assigned = True
                        break
            
            if not assigned:
                onset_groups[note['onset']].append(note)
        
        # Create symbols from the groups
        for onset_time, note_group in sorted(onset_groups.items()):
            if len(note_group) == 1:
                # Single note
                note = note_group[0]
                symbols.append({
                    'type': 'note',
                    'pitch': note['pitch'],
                    'duration': int(note['duration']),
                    'velocity': note['velocity'],
                    'onset': note['onset']
                })
            else:
                # Chord
                pitches = [note['pitch'] for note in note_group]
                durations = [note['duration'] for note in note_group]
                velocities = [note['velocity'] for note in note_group]
                
                # Calculate the total duration as the time from first onset to last offset
                offsets = [note['onset'] + note['duration'] for note in note_group]
                total_duration = max(offsets) - onset_time
                
                symbols.append({
                    'type': 'chord',
                    'pitch': pitches,
                    'duration': int(total_duration),
                    'velocity': int(np.mean(velocities)),
                    'onset': onset_time
                })
        
        return symbols

    def process_midi_file(self, midi_file: str) -> List[Dict]:
        """
        Process a MIDI file to extract symbols.
        
        Args:
            midi_file: Path to the MIDI file
            
        Returns:
            List of symbols (notes and chords)
        """
        notes = self.extract_notes(midi_file)
        symbols = self.create_symbols(notes)
        return symbols

    def serialize_symbols(self, symbols: List[Dict]) -> str:
        """
        Serialize symbols to JSON for efficient storage/transfer.
        
        Args:
            symbols: List of symbol dictionaries
            
        Returns:
            JSON string representation
        """
        return json.dumps(symbols)
    
    def deserialize_symbols(self, json_str: str) -> List[Dict]:
        """
        Deserialize symbols from JSON.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            List of symbol dictionaries
        """
        return json.loads(json_str)

    def process_dataset(self, dataset_dir: str, 
                        output_file: Optional[str] = None, 
                        transpose: bool = False, 
                        transposition_interval: int = 12) -> List[Dict]:
        """
        Process all MIDI files in a dataset directory into one flat list of symbols.

        Args:
            dataset_dir: Directory containing MIDI files
            output_file: Optional file path to save the processed dataset (as a JSON list)
            transpose: Optional bool to transpose the dataset in every ton
        Returns:
            List of symbol dictionaries (notes and chords) from all files
        """
        midi_files = self.find_midi_files(dataset_dir)
        all_symbols: List[Dict] = []

        for midi_file in midi_files:
            # Extract symbols for this file
            symbols = self.process_midi_file(midi_file)
            #—and append them to the single, global list
            all_symbols.extend(symbols)
        if transpose:
           for i in range(transposition_interval):
            transposed = self.transpose_symbols(symbols, i+1)
            all_symbols.extend(transposed) 
        # Optionally, write out the unified list
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_symbols, f)

        return all_symbols
    
    def transpose_symbols(self, symbols: List[Dict], semitone_shift: int) -> List[Dict]:
        """
        Transpose a list of symbols by a number of semitones.
        """
        transposed = []
        for sym in symbols:
            new_sym = copy.deepcopy(sym)
            if new_sym['type'] == 'note':
                new_sym['pitch'] += semitone_shift
            else:
                new_sym['pitch'] = [p + semitone_shift for p in new_sym['pitch']]
            transposed.append(new_sym)
        return transposed
    
    def create_notesequence_tensor(self, midi_file: str, randomize_chord_order: bool = False) -> np.ndarray:
        """
        Create NoteSequence tensor compatible with Piano Genie data format.
        
        Args:
            midi_file: Path to the MIDI file
            randomize_chord_order: If True, randomize order of simultaneous notes
            
        Returns:
            Numpy array with shape [num_notes, 5] containing:
            [pitch, velocity, delta_time, start_time, end_time]
        """
        notes = self.extract_notes(midi_file)
        if not notes:
            return np.empty((0, 5), dtype=np.float32)
        
        # Convert to seconds for consistency with Piano Genie format
        note_data = []
        for note in notes:
            note_data.append({
                'pitch': note['pitch'],
                'velocity': note['velocity'],
                'start_time': note['onset'] / 1000.0,  # Convert ms to seconds
                'end_time': (note['onset'] + note['duration']) / 1000.0
            })
        
        # Sort by start_time, then by pitch (Piano Genie standard)
        if randomize_chord_order:
            # Group by start time for chord randomization
            time_groups = defaultdict(list)
            for note in note_data:
                # Quantize to nearest millisecond for grouping
                time_key = round(note['start_time'] * 1000)
                time_groups[time_key].append(note)
            
            # Randomize within each time group, then sort groups by time
            sorted_notes = []
            for time_key in sorted(time_groups.keys()):
                group = time_groups[time_key]
                np.random.shuffle(group)
                # Sort by start_time within shuffled group for stability
                group.sort(key=lambda n: n['start_time'])
                sorted_notes.extend(group)
        else:
            # Standard sorting: start_time, then pitch
            sorted_notes = sorted(note_data, key=lambda n: (n['start_time'], n['pitch']))
        
        # Filter piano range [21, 108] as in Piano Genie
        filtered_notes = [n for n in sorted_notes if 21 <= n['pitch'] <= 108]
        
        if not filtered_notes:
            return np.empty((0, 5), dtype=np.float32)
        
        # Calculate delta times
        pitches = np.array([n['pitch'] for n in filtered_notes])
        velocities = np.array([n['velocity'] for n in filtered_notes])
        start_times = np.array([n['start_time'] for n in filtered_notes])
        end_times = np.array([n['end_time'] for n in filtered_notes])
        
        # Delta times: difference between consecutive note starts
        if len(start_times) > 1:
            # First note gets large delta (100000.0) as in Piano Genie
            delta_times = np.concatenate([[100000.0], start_times[1:] - start_times[:-1]])
        else:
            delta_times = np.array([100000.0])
        
        # Stack into tensor format: [pitch, velocity, delta_time, start_time, end_time]
        tensor = np.stack([pitches, velocities, delta_times, start_times, end_times], axis=1).astype(np.float32)
        
        return tensor
    
    def process_dataset_to_notesequences(self, dataset_dir: str, 
                                       output_file: Optional[str] = None,
                                       randomize_chord_order: bool = False) -> List[np.ndarray]:
        """
        Process all MIDI files in a dataset directory to NoteSequence tensors.
        
        Args:
            dataset_dir: Directory containing MIDI files
            output_file: Optional file path to save tensors (as .npz)
            randomize_chord_order: If True, randomize chord note order
            
        Returns:
            List of NoteSequence tensors (one per MIDI file)
        """
        midi_files = self.find_midi_files(dataset_dir)
        notesequences = []
        
        for midi_file in midi_files:
            tensor = self.create_notesequence_tensor(midi_file, randomize_chord_order)
            if tensor.shape[0] > 0:  # Only add non-empty sequences
                notesequences.append(tensor)
        
        if output_file:
            # Save as compressed numpy format
            np.savez_compressed(output_file, *notesequences)
        
        return notesequences
    
    @staticmethod
    def demidify_tensor(tensor: np.ndarray) -> np.ndarray:
        """
        Transform MIDI pitches [21,108] to [0,87] in NoteSequence tensor.
        
        Args:
            tensor: NoteSequence tensor with MIDI pitches in first column
            
        Returns:
            Tensor with transformed pitches
        """
        result = tensor.copy()
        result[:, 0] = tensor[:, 0] - 21  # pitch column
        return result
    
    @staticmethod
    def remidify_tensor(tensor: np.ndarray) -> np.ndarray:
        """
        Transform [0,87] to MIDI pitches [21,108] in NoteSequence tensor.
        
        Args:
            tensor: NoteSequence tensor with transformed pitches in first column
            
        Returns:
            Tensor with MIDI pitches
        """
        result = tensor.copy()
        result[:, 0] = tensor[:, 0] + 21  # pitch column
        return result
