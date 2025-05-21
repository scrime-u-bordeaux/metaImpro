"""
Fichier qui a pour but d'évaluer la démarche scientifique et de l'orienter
Plusieurs idées : 
1. Fixer le geste et les modèles : permet d'évaluer l' impact du hasard,  d'un geste donnée pour un modèle donnée
2. Faire une fonction qui au fil du temps de la note récupère les probabilités de la note choisie et trace le point correspondant 
à la probabilitée en fonction du temps.
3. Faire un piano roll pour la data visualisation.
"""
from typing import List
import matplotlib.pyplot as plt
import json, os, re

EVAL_P_DIR = "eval/probs"
EVAL_G_DIR = "eval/graph"
BASENAME = "probs"
EXT = ".json"


def probability_plot(filename: str):
    """
    Fonction qui récupère une liste de probs et qui les tracent (abscisse : probs, ordonnée : limite de la distribution [0:1])

    Args:
        probs (List[float]): liste des probabilités
    """
    open_path = os.path.join(EVAL_P_DIR, filename)
    with open(open_path, "r") as f:
        prob_history = json.load(f)
    # Création du graphe
    plt.figure()
    plt.title("Historique des probabilités")
    plt.xlabel("Index de l'événement")
    plt.ylabel("Probabilité")
    plt.ylim(0, 1)  # Limites de l'axe Y entre 0 et 1
    plt.plot(prob_history, marker='o')  # on trace prob_history pour être cohérent

    # Chemin de sauvegarde dans eval/
    base = os.path.splitext(filename)[0]
    save_path = os.path.join(EVAL_G_DIR, f"{base}.png")
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
    # chemin vers un json de probs déjà présent dans eval/
    json_path = "probs_001_Syeeda's Song Flute - John Coltrane Transcription.json"
    probability_plot(json_path)