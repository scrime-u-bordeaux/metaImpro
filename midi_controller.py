from collections import Counter
from typing import Dict, Any
import numpy as np
import warnings
import os
from midi_processor import MidiSymbolProcessor
import markov as mv


class Distribution:

    def __init__(self, path):
        self.path  = path
        eval_dir = "/eval"

    def get_distrib(self):
        if self.path.lower().endswith(('.mid', '.midi')):
            proc = MidiSymbolProcessor()
            symbols = proc.find_midi_files(self.path)
            all_keys = [mv.symbol_to_key(s) for s in symbols]
            init_pitches = [key[1] for key in all_keys if isinstance(key, tuple) and len(key) > 1]
            counts = Counter(init_pitches)
            init_distribution = {pitch: count for pitch, count in counts.items()}   

        elif self.path.lower().endswith('.json'):
            return 2
        else:
            raise ValueError(
                f"Unsupported file extension {os.path.splitext(self.path)[1]!r}; "
                "must be .mid, .midi or .json"
            )
    def normalize(self):
        return
    
    def distance(self):
        return
    
    def plot_histogram(self):
        return
    
    def plot_density(self):
        #seaborne kde
        return
    
d = Distribution("Histogram_probs_001_MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_08_Track08_wav.midi")
d.get_distrib()