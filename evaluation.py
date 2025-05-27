"""
Fichier qui a pour but d'évaluer la démarche scientifique et de l'orienter
Plusieurs idées : 
1. Fixer le geste et les modèles : permet d'évaluer l' impact du hasard,  d'un geste donnée pour un modèle donnée
2. Faire une fonction qui au fil du temps de la note récupère les probabilités de la note choisie et trace le point correspondant 
à la probabilitée en fonction du temps.
3. Faire un piano roll pour la data visualisation.
"""
import markov as mv
import matplotlib.pyplot as plt
import json, os, re
from collections import Counter
from midi_processor import MidiSymbolProcessor
import ast
import numpy as np
from scipy.stats import entropy, wasserstein_distance

EVAL_P_DIR = "eval/probs"
EVAL_G_DIR = "eval/graph"
EVAL_H_DIR = "eval/histogram"
BASENAME = "probs"
EXT = ".json"

def fixed_gesture():
    """
    temps d'appui, nb monté/descente
    """

def distance_hist(dist1, dist2):

    all_bins = sorted(set(dist1) | set(dist2))
    counts1 = np.array([dist1.get(b, 0) for b in all_bins], dtype=float)
    counts2 = np.array([dist2.get(b, 0) for b in all_bins], dtype=float)

    # 2) Normalize to probabilities
    p = counts1 / counts1.sum()
    q = counts2 / counts2.sum()

    # 3) Compute distances
    l1 = np.linalg.norm(p - q, ord=1)
    l2 = np.linalg.norm(p - q, ord=2)

    # Jensen–Shannon
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    # Earth Mover’s (only makes sense on integer bins)
    try:
        # if all_bins are integers or can be cast
        positions = np.array([int(b) for b in all_bins])
        emd = wasserstein_distance(positions, positions,
                                   u_weights=counts1,
                                   v_weights=counts2)
    except ValueError:
        emd = None

    return {
        'L1': l1,
        'L2': l2,
        'JS': js,
        'EMD': emd
    }
    
def parse_pitch(pitch_str):
    try:
        val = ast.literal_eval(pitch_str)
        return tuple(val) if isinstance(val, list) else int(val)
    except Exception:
        return pitch_str  # fallback if malformed

def get_distrib(midi_path, json_path):
    processor = MidiSymbolProcessor()
    symbols =processor.process_midi_file(midi_path)
    all_keys = [mv.symbol_to_key(s) for s in symbols]
    init_pitches = [key[1] for key in all_keys if isinstance(key, tuple) and len(key) > 1]
    counts = Counter(init_pitches)
    init_distribution = {pitch: count for pitch, count in counts.items()}   

    open_path = os.path.join(EVAL_P_DIR, json_path)
    with open(open_path, 'r') as f:
        data = json.load(f)
    pitches = [parse_pitch(pitch_str) for pitch_str, _ in data]
    empirical_distribution = Counter(pitches)

    return init_distribution, empirical_distribution

def histogram_plot(init_distribution, empirical_distribution, ):

    x_axes = ["Pitch", "Pitch"]
    y_axes = ["Count", "Count"]
    titles = ["Initial Distribution", "Generated Distribution"]

    pitches = [str(p) for p in init_distribution.keys()]
    probs = list(init_distribution.values())
    f, a = plt.subplots(2, 1, figsize=(20, 10))

# Prepare data for plotting
    distributions = [init_distribution, empirical_distribution]

    for idx, ax in enumerate(a):
        dist = distributions[idx]
        int_keys = [k for k in dist.keys() if isinstance(k, int)]
        tuple_keys = [k for k in dist.keys() if isinstance(k, tuple)]
        sorted_keys = sorted(int_keys) + tuple_keys

        # Prepare for plotting
        pitches = [str(k) for k in sorted_keys]
        probs = [dist[k] for k in sorted_keys]

        ax.bar(pitches, probs, color='skyblue')
        ax.set_title(titles[idx])
        ax.set_xlabel(x_axes[idx])
        ax.set_ylabel(y_axes[idx])


    base = os.path.splitext(json_path)[0]
    save_path = os.path.join(EVAL_H_DIR, f"Histogram_{base}.png")
    plt.savefig(save_path)
    plt.close()

def probability_plot(filename: str, mode:str="plot", correction=False):
    """
    Fonction qui récupère une liste de probs et qui les tracent (abscisse : probs, ordonnée : limite de la distribution [0:1])

    Args:
        probs (List[float]): liste des probabilités
    """
    open_path = os.path.join(EVAL_P_DIR, filename)
    with open(open_path, "r") as f:
        prob_history = json.load(f)

    if correction:
        prob_history = [prob[1] for prob in prob_history if prob[1] !=1]
        corrected = "_corrected"
    else:
        prob_history = [prob[1] for prob in prob_history]
        corrected = "_"

    # Création du graphe
    plt.figure()
    plt.title("Historique des probabilités")
    plt.xlabel("Index de l'événement")
    plt.ylabel("Probabilité")
    plt.ylim(-0.1, 1.1)  # Limites de l'axe Y entre 0 et 1
    
    x = list(range(len(prob_history)))
    y = prob_history
    if mode == "plot":
        plt.plot(x, y, marker='o')
    elif mode == "scatter":
        plt.scatter(x, y, marker='o')
    else:
        raise ValueError(f"Mode inconnu : {mode!r}. Utilise 'plot' ou 'scatter'.")

    # Chemin de sauvegarde dans eval/
    base = os.path.splitext(filename)[0]
    
    save_path = os.path.join(EVAL_G_DIR, f"{mode}_{base}{corrected}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Graphe sauvegardé dans : {save_path}")


def save_prob_history_incremental(prob_history, title: str):

    os.makedirs(EVAL_P_DIR, exist_ok=True)

    # Nettoie le titre pour enlever toute extension
    title_clean = os.path.splitext(title)[0]

    # Récupère les indices existants
    existing = os.listdir(EVAL_P_DIR)
    pattern = re.compile(rf"^{BASENAME}_(\d{{3}}){re.escape(EXT)}$")
    indices = [
        int(m.group(1))
        for f in existing
        if (m := pattern.match(f))
    ]

    next_idx = max(indices) + 1 if indices else 1
    filename = f"{BASENAME}_{next_idx:03d}_{title_clean}{EXT}"
    path = os.path.join(EVAL_P_DIR, filename)

    with open(path, "w") as fp:
        json.dump(prob_history, fp, indent=2)

    print(f"Saved {len(prob_history)} probs into {path}")
    return path


#PLOT LA PROB

if __name__ == "__main__":
    midi_path = "corpus/MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_08_Track08_wav.midi"
    json_path = "probs_001_MIDI-Unprocessed_02_R1_2011_MID--AUDIO_R1-D1_08_Track08_wav.json"
    histogram_plot(midi_path, json_path)     
    for _, _, files in os.walk(EVAL_P_DIR):
        for name in files:    
            probability_plot(name, "plot", False)