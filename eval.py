from collections import Counter
import json
import os
import numpy as np
from midi_processor import MidiSymbolProcessor
import matplotlib.pyplot as plt
import seaborn as sn

class Eval:

    def __init__(self, path:str):
        self.path  = path
        self.hist_dir = "eval/histogram/"
        self.graph_dir = "eval/graph"
        self.pitches, self.probs = self.load_file()

    def load_file(self):
        if self.path.lower().endswith(('.mid', '.midi')):
            processor = MidiSymbolProcessor()
            symbols = processor.process_midi_file(self.path)
            raw = [pitch['pitch'] for pitch in symbols]
            self.probs = None
        elif self.path.lower().endswith('.json'):
            with open(self.path, 'r') as f:
                data = json.load(f)
            raw = [pitch[0] for pitch in data]
            self.probs = [prob[1] for prob in data]

        else:
            raise ValueError(
                f"Unsupported file extension {os.path.splitext(self.path)[1]!r}; "
                "must be .mid, .midi or .json"
            )
        self.pitches = self.flatten(raw)
        return self.pitches, self.probs
    
    def get_distrib(self):
        count = Counter(self.pitches)
        return count
    
    def normalize(self, data):
        normalized = [tuple(x) if isinstance(x, list) else x for x in data]
        return normalized
    
    def flatten(self, data):
        flat = []
        for x in data:
            if isinstance(x, (list, tuple)):
                flat.extend(x)
            else:
                flat.append(x)
        return flat

    def _aligned_probs(self, other: "Eval"):
        """Retourne deux vecteurs de probabilités alignés sur l'union des clés."""
        dist1 = self.get_distrib()
        dist2 = other.get_distrib()
        # union des clés
        keys = sorted(set(dist1.keys()) | set(dist2.keys()), key=lambda x: str(x))
        # compter totaux
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        # vecteurs
        p = np.array([dist1.get(k, 0) / total1 for k in keys], dtype=float)
        q = np.array([dist2.get(k, 0) / total2 for k in keys], dtype=float)
        return p, q
    
    def distance(self, other: "Eval", type_dist="euclidean"):

        p, q = self._aligned_probs(other)

        if type_dist == "euclidean":
            return np.linalg.norm(p - q)
    
    def plot_notes(self):
        # Vérification de base
        if not isinstance(self.probs, list) or len(self.probs) == 0:
            raise ValueError(
                "You must use a non-empty list of [pitch, probability] pairs in self.probs"
            )
        
        # Préparation des données x et y
        x = list(range(len(self.probs)))
        y = []
        
        for idx, item in enumerate(self.probs):
           y.append(item)
        
        # Trace
        os.makedirs(self.graph_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, linestyle='-')
        plt.xlabel("Index de l'événement")
        plt.ylabel("Probabilité")
        plt.title("Probabilité en fonction de l'index de l'événement")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        
        # Sauvegarde
        base = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(self.graph_dir, f"Prob_vs_Index_{base}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot successfully saved in {save_path}")



    def plot_histogram(self):
        count = self.get_distrib()
        # Convert pitch keys to strings (to handle tuples cleanly)
        labels = sorted(count.keys())
        values = list(count.values())

        os.makedirs(self.hist_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.xlabel("Probabilité")
        plt.ylabel("Count")
        plt.title("Distribution des notes dans le morceau")
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        plt.tight_layout()

        base = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(self.hist_dir + f"Histogram_{base}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Histogram successfully saved in {self.hist_dir}")
            
    def plot_density(self):

        # Récupère les données brutes (uniquement des int après flatten)
        raw_pitches = [p for p in self.pitches if isinstance(p, int)]

        if not raw_pitches:
            print("No valid integer pitch data for density plot.")
            return

        # Crée le dossier d'histogrammes s'il n'existe pas
        os.makedirs(self.hist_dir, exist_ok=True)

        # Trace la KDE
        plt.figure(figsize=(12, 6))
        sn.kdeplot(raw_pitches, fill=True)
        plt.xlabel("Pitch")
        plt.ylabel("Density")
        plt.title("Distribution des notes dans le morceau")
        plt.tight_layout()

        # Prépare et sauve le fichier
        base = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(self.hist_dir, f"Density_plot_{base}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Density plot successfully saved in {save_path}")
    
    

path1 = "eval/probs/probs_001_MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--1.json"   
path2 = "corpus/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_02_R1_2015_wav--1.midi"
e1 = Eval(path1)
e2 = Eval(path2)
e1.plot_notes()
e2.plot_notes()
#e1.plot_histogram()
#e2.plot_histogram()
#e1.plot_density()
#e2.plot_density()
#e1.plot_notes()
#print("Euclidienne:", e1.distance(e2))

