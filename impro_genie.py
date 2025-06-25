import torch
import torch.nn.functional as F
import numpy as np
import json
import pathlib
from typing import Dict, Any, List, Tuple, Optional
import model as md

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
            # Default configuration matching your training setup
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
        
        # State management
        self.decoder_hidden = None
        self.button_history = []
        self.time_history = []
        self.note_history = []
        self.max_context = 32