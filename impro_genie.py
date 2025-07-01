import torch
import torch.nn.functional as F
import numpy as np
import json
import pathlib
from typing import Dict, Any, List, Tuple, Optional
import model as md
import time

class PianoGenieEngine:
    """
    Piano Genie improvisation engine
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the Piano Genie engine.
        
        Args:
            model_path: Path to the trained .pt model file
            config_path: Optional path to config.json file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path and pathlib.Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.cfg = json.load(f)
        else:
            # Default configuration
            self.cfg = {
                "num_buttons": 8,
                "model_rnn_dim": 128,
                "model_rnn_num_layers": 2,
                "seq_len": 128,
                "data_delta_time_max": 1.0,
            }
        # Load the trained model
        self.model = md.PianoGenieAutoencoder(self.cfg).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def reset_generation(self):
        """À appeler au démarrage pour réinitialiser l’état du décodeur."""
        self.k_prev = torch.full((1,1), md.SOS, dtype=torch.long, device=self.device)
        self.h, self.c = self.model.dec.init_hidden(1, device=self.device)
        self.last_time = time.time()

    def generate_note_from_button(self, button_idx: int, delta_t: float):
        """
        Appelé à chaque appui de bouton (0…7).
        Renvoie (midi_pitch, onset_time).
        """
        # clamp delta
        delta_t = min(delta_t, self.cfg["data_delta_time_max"])
        
        # tensors
        k_in = self.k_prev                  # (1,1)
        t_in = torch.tensor([[delta_t]], device=self.device)
        b_in = torch.tensor([[button_idx]], device=self.device)
        
        logits, (h_new, c_new) = self.model.dec(k_in, t_in, b_in, (self.h, self.c))
        probs      = F.softmax(logits[0,0], dim=-1)
        pitch_idx  = torch.multinomial(probs, 1).item()
        
        # update state
        self.k_prev = torch.tensor([[pitch_idx]], device=self.device)
        self.h, self.c = h_new, c_new

        # compute midi pitch
        midi_pitch = pitch_idx + md.PIANO_LOWEST_KEY_MIDI_PITCH

        # compute onset time relative to now (or relative to last_time if you prefer)
        onset_time = time.time()
        # update last_time if you want to measure deltas externally
        self.last_time = onset_time

        return midi_pitch, onset_time