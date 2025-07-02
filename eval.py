from collections import Counter
import json
import os
import numpy as np
from midi_processor import MidiSymbolProcessor
import matplotlib.pyplot as plt
import seaborn as sns

class Eval:

    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, path: str):
        self.path = path
        self.hist_dir = "eval/histogram/"
        self.graph_dir = "eval/graph/"
        self.pitches, self.probs = self.load_file()

    def load_file(self):
        if self.path.startswith("corpus"):
            return self._load_corpus()
        elif self.path.startswith("eval"):
            return self._load_markov()
        elif self.path.startswith("piano_genie"):
            return self._load_dataset()
        else:
            ext = os.path.splitext(self.path)[1]
            raise ValueError(
                f"Unsupported path {self.path!r} (ext {ext!r}); "
                "must start with corpus/, eval/ or piano_genie/."
            )

    def _load_corpus(self):
        """Charge un unique fichier MIDI ou JSON de symboles."""
        if self.path.lower().endswith(('.mid', '.midi')):
            proc = MidiSymbolProcessor()
            symbols = proc.process_midi_file(self.path)
            raw = [p['pitch'] for p in symbols]
        else:
            with open(self.path, 'r') as f:
                data = json.load(f)
            raw = [p['pitch'] for p in data]
        return self.flatten(raw), None

    def _load_markov(self):
        """Charge un JSON de paires [pitch, prob] produit par ton Markov."""
        with open(self.path, 'r') as f:
            data = json.load(f)
        raw = [item[0] for item in data]
        probs = [item[1] for item in data]
        return self.flatten(raw), probs

    def _load_dataset(self):
        """
        Charge le cache piano_genie/..._performances.json,
        extrait tous les pitch_idx (3ᵉ élément de chaque tuple).
        """
        with open(self.path, 'r') as f:
            performances = json.load(f)
        flat = []
        for perf in performances:
            # perf est une liste de (onset, dur, pitch_idx, vel)
            flat.extend([entry[2] for entry in perf])
        return flat, None
    
    def get_distrib(self):
        count = Counter(self.pitches)
        return count
    
    def num2note(self, pitch_num: int) -> str:
        """Convert MIDI pitch number to note name (ignore octave)."""
        return self.NOTES[pitch_num % 12]
    
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
        keys = sorted(set(dist1.keys()) | set(dist2.keys()))
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
        else:
            raise ValueError(f"Distance type {type_dist} not implemented")
    
    def plot_notes(self):
        # Vérification de base
        if self.probs is None or len(self.probs) == 0:
            raise ValueError(
                "You must use a non-empty list of probabilities in self.probs. "
                "This method only works with Markov-generated data."
            )
        
        # Préparation des données x et y
        x = list(range(len(self.probs)))
        y = self.probs
        
        # Trace
        os.makedirs(self.graph_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, marker='.', linestyle='-')
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
        labels = [str(k) for k in sorted(count.keys())]
        values = [count[k] for k in sorted(count.keys())]

        os.makedirs(self.hist_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.xlabel("Pitch")
        plt.ylabel("Count")
        plt.title("Distribution des notes dans le morceau")
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        plt.tight_layout()

        base = os.path.splitext(os.path.basename(self.path))[0]
        save_path = os.path.join(self.hist_dir, f"Histogram_{base}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Histogram successfully saved in {save_path}")
            
    def plot_density(self):
        # Récupère les données brutes (uniquement des int après flatten)
        raw_pitches = [p for p in self.pitches if isinstance(p, (int, float))]

        if not raw_pitches:
            print("No valid numeric pitch data for density plot.")
            return

        # Crée le dossier d'histogrammes s'il n'existe pas
        os.makedirs(self.hist_dir, exist_ok=True)

        # Trace la KDE
        plt.figure(figsize=(12, 6))
        sns.kdeplot(raw_pitches, fill=True)
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

    def compute_perplexity(self) -> float:
        """
        Calcule la perplexité pour la séquence de probabilités chargée (self.probs).
        Perplexité: exp(-1/N * sum(log p_i)).
        """
        if self.probs is None or len(self.probs) == 0:
            raise ValueError("Aucune probabilité disponible pour calculer la perplexité.")
        
        # Éviter les log(0) en filtrant ou en ajoutant un epsilon
        safe_probs = [max(p, 1e-10) for p in self.probs]
        log_probs = np.log(safe_probs)
        N = len(log_probs)
        return float(np.exp(-log_probs.sum() / N))


if __name__ == "__main__":
    path1 = "eval/probs/probs_001_bach_chorales.json"   
    path2 = "corpus/bach_chorales.json"
    path3 = "piano_genie/maestro-v3.0.0-midi_performances.json"
    
    e1 = Eval(path1)  # Données Markov avec probabilités
    e2 = Eval(path2)  # Données corpus sans probabilités
    e3 = Eval(path3)  # Données dataset piano_genie sans probabilités
    
    # Plots spécifiques selon le type de données
    if e1.probs is not None:
        e1.plot_notes()  # Seulement pour les données avec probabilités
        print("Perplexité e1:", e1.compute_perplexity())
    
    # Plots disponibles pour tous types de données
    print("Generating plots...")
    e1.plot_histogram()
    e2.plot_histogram()
    e3.plot_histogram()  # Histogramme du dataset piano_genie
    
    e1.plot_density()
    e2.plot_density()
    e3.plot_density()    # Densité du dataset piano_genie
    
    # Comparaisons de distances
    print("Distance euclidienne e1-e2 (Markov vs Corpus):", e1.distance(e2))
    print("Distance euclidienne e1-e3 (Markov vs Dataset):", e1.distance(e3))
    print("Distance euclidienne e2-e3 (Corpus vs Dataset):", e2.distance(e3))
    
    # Statistiques
    print(f"Nombre de notes - e1: {len(e1.pitches)}, e2: {len(e2.pitches)}, e3: {len(e3.pitches)}")
    print(f"Range de pitches - e1: {min(e1.pitches)}-{max(e1.pitches)}, "
          f"e2: {min(e2.pitches)}-{max(e2.pitches)}, "
          f"e3: {min(e3.pitches)}-{max(e3.pitches)}")