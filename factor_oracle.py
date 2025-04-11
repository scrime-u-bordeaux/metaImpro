import os
import json
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Union

class FactorOracle:
    """
    Oracle des facteurs pour l’improvisation musicale.
    Utilise NumPy pour améliorer les performances.
    Basé sur l'algorithme de Assayag et Dubnov.
    """
    def __init__(self):
        """Initialisation de l'oracle des facteurs"""
        self.states = [{}]  # État 0 initial
        self.transitions = [{} for _ in range(1)]  # Transitions sortantes pour chaque état
        self.suffix_links = np.array([None], dtype=object)  # Liens suffixes
        self.lrs = np.zeros(1, dtype=np.int32)  # Longueur du suffixe répété le plus long
        self.symbols = [None]  # Symboles associés à chaque état
        self.file_names = []  # List to store the names of files used

    def add_symbol(self, symbol: Dict[str, Any]) -> None:
        """
        Ajoute un symbole et met à jour transitions et liens suffixes.
        Le symbole contient par exemple pitch, duration, velocity, in_chord, chord_id, etc.
        """
        symbol_hash = self._get_symbol_hash(symbol)
        new_state_index = len(self.states)
        self.states.append({})
        self.transitions.append({})
        self.suffix_links = np.append(self.suffix_links, None)
        self.lrs = np.append(self.lrs, 0)
        self.symbols.append(symbol)
        self.transitions[new_state_index-1][symbol_hash] = new_state_index
        k = self.suffix_links[new_state_index-1]
        while k is not None and symbol_hash not in self.transitions[k]:
            self.transitions[k][symbol_hash] = new_state_index
            k = self.suffix_links[k]
        if k is None:
            self.suffix_links[new_state_index] = 0
        else:
            self.suffix_links[new_state_index] = self.transitions[k][symbol_hash]
        if self.suffix_links[new_state_index] == 0:
            self.lrs[new_state_index] = 0
        else:
            self.lrs[new_state_index] = self.lrs[self.suffix_links[new_state_index]] + 1

    def _get_symbol_hash(self, symbol: Dict[str, Any]) -> Tuple:
        """
        Crée une version hashable du symbole, en conservant les informations essentielles.
        """
        return (
            symbol.get('pitch', -1),
            symbol.get('duration', -1),
            symbol.get('velocity', -1),
            symbol.get('in_chord', False),
            symbol.get('chord_id', None),
            symbol.get('duration_category', 'inconnu')
        )

    def learn_sequence(self, symbols: List[Dict[str, Any]]) -> None:
        """
        Apprend une séquence complète de symboles.
        """
        total_symbols = len(symbols)
        if total_symbols > 0:
            current_size = len(self.states)
            new_size = current_size + total_symbols
            new_suffix_links = np.empty(new_size, dtype=object)
            new_suffix_links[:current_size] = self.suffix_links
            new_suffix_links[current_size:] = None
            self.suffix_links = new_suffix_links
            new_lrs = np.zeros(new_size, dtype=np.int32)
            new_lrs[:current_size] = self.lrs
            self.lrs = new_lrs
            for symbol in symbols:
                self.add_symbol(symbol)

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        n_states = len(self.states)
        n_transitions = sum(len(t) for t in self.transitions)
        max_suffix_link_length = int(np.max(self.lrs)) if n_states > 0 else 0
        avg_outgoing_transitions = n_transitions / n_states if n_states > 0 else 0
        return {
            "number_of_states": n_states,
            "number_of_transitions": n_transitions,
            "max_suffix_link_length": max_suffix_link_length,
            "avg_outgoing_transitions": avg_outgoing_transitions
        }

    def save(self, filename: str) -> None:
        """
        Sauvegarde l'oracle dans un fichier JSON, y compris les noms des fichiers utilisés.
        """
        data = {
            "states": len(self.states),
            "symbols": self.symbols,
            "transitions": [{str(k): v for k, v in state_trans.items()} for state_trans in self.transitions],
            "suffix_links": self.suffix_links.tolist(),
            "lrs": self.lrs.tolist(),
            "file_names": self.file_names  # Include file names in the JSON
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'FactorOracle':
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        oracle = cls()
        oracle.states = [{} for _ in range(data["states"])]
        oracle.symbols = data["symbols"]
        oracle.suffix_links = np.array(data["suffix_links"], dtype=object)
        oracle.lrs = np.array(data["lrs"], dtype=np.int32)
        oracle.file_names = data.get("file_names", [])  # Load file names if available
        oracle.transitions = []
        for state_trans in data["transitions"]:
            transitions = {}
            for k_str, v in state_trans.items():
                k = eval(k_str)
                transitions[k] = v
            oracle.transitions.append(transitions)
        return oracle

def process_midi_data_with_oracle(json_file: str, output_dir: Optional[str] = None, limit: Optional[int] = None) -> Tuple[FactorOracle, int]:
    """
    Charge un fichier JSON contenant les symboles extraits (par exemple via un process_dataset)
    et construit un oracle en choisissant aléatoirement un sous-ensemble des fichiers.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    oracle = FactorOracle()
    file_count = 0
    symbol_count = 0
    all_symbols = []
    files = data['files']
    random.shuffle(files)  # Files are chosen in random order
    for file_data in files:
        if 'separator' in file_data:
            continue
        if limit is not None and file_count >= limit:
            break
        if 'symbols' in file_data:
            all_symbols.extend(file_data['symbols'])
            file_count += 1
            symbol_count += len(file_data['symbols'])
            oracle.file_names.append(file_data.get('file_name', f"file_{file_count}"))  # Store file name
    oracle.learn_sequence(all_symbols)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        oracle.save(os.path.join(output_dir, f"factor_oracle_{limit if limit is not None else 'all'}.json"))
    return oracle, symbol_count

if __name__ == "__main__":
    input_file = "FO/all_symbols.json"
    output_dir = "FO/oracle_results"
    oracle, symbol_count = process_midi_data_with_oracle(
        input_file,
        output_dir=output_dir,
        limit=1
    )
    print(f"Oracle des Facteurs créé avec succès! {symbol_count} symboles traités.")
