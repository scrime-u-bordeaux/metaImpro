import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional
from accompaniement import get_pitches_by_chord


def build_interval_dict(chord_map):
    interval_dict = {}
    for chord, pitches in chord_map.items():
        pitches = list(pitches)
        intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
        interval_dict[chord] = intervals
    return interval_dict

def build_interval_table(data, max_order):
    table = defaultdict(lambda: defaultdict(int))
    for seq in data.values():
        seq = list(seq)
        for i in range(len(seq)):
            succ = seq[i]
            for order in range(1, max_order + 1):
                if i - order < 0:
                    break
                ctx = tuple(seq[i-order:i])
                table[ctx][succ] += 1
    return {ctx: dict(d) for ctx, d in table.items()}


def generate_symbols_intervals(
    previous_intervals,    # int ou liste d'int : historique récent d'intervalles
    inter_table,           # table de transitions: contexte -> {interval: count}
    order=1,               # ordre max à utiliser pour le contexte
    n_candidates=1,        # nombre de candidats parmi les plus probables à considérer
    base_pitch=None,       # si fourni, on calcule next_pitch = base_pitch + chosen_interval
    chord_scale=None,      # liste de pitches possibles pour fallback (ex: gamme de l'accord)
    seed=None              # optionnel : rendre le tirage reproductible
):
    """
    Retour:
        chosen_interval : intervalle choisi (int)
        prob            : probabilité associée (float)
        top_candidates  : liste des (interval, prob) les plus probables (max 4)
        next_pitch      : base_pitch + chosen_interval si base_pitch donné, sinon None
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1) Normaliser l'historique en liste d'entiers
    if isinstance(previous_intervals, int):
        history = [previous_intervals]
    elif previous_intervals is None:
        history = []
    else:
        history = list(previous_intervals)

    # 2) Construire le contexte maximal (dernier 'order' éléments)
    context = tuple(history[-order:]) if history else ()

    # util interne: choix parmi top-k (on choisit uniform parmi top n_candidates)
    def pick_from_counts(keys, counts, top_k):
        counts = np.array(counts, dtype=float)
        if counts.sum() <= 0:
            return None, None, None
        probs = counts / counts.sum()
        order_idx = np.argsort(probs)[::-1]
        top_k = min(top_k, len(keys))
        candidates_idx = order_idx[:top_k]
        chosen_pos = int(np.random.choice(candidates_idx))
        chosen_key = keys[chosen_pos]
        chosen_prob = float(probs[chosen_pos])
        top_idx = order_idx[:min(4, len(keys))]
        top_probs = [(keys[i], float(probs[i])) for i in top_idx]
        return chosen_key, chosen_prob, top_probs

    # 3) Back-off : chercher le plus long suffixe du contexte présent dans la table
    dist = None
    found_context = None
    for backoff in range(min(order, len(context)), 0, -1):
        sub = context[-backoff:]
        if sub in inter_table:
            dist = inter_table[sub]
            found_context = sub
            break

    # 4) Si pas trouvé, fallback marginal (regarder l'ensemble des successeurs)
    if dist is None:
        marg_counts = defaultdict(float)
        for sub_dist in inter_table.values():
            for iv, cnt in sub_dist.items():
                marg_counts[iv] += cnt
        if len(marg_counts) == 0:
            # table vide : fallback vers chord_scale si fournie, sinon 0 interval
            if chord_scale:
                chosen_interval = random.choice([p - (base_pitch if base_pitch is not None else chord_scale[0]) for p in chord_scale])
                prob = 1.0
                top_candidates = [(chosen_interval, 1.0)]
                next_pitch = (base_pitch + chosen_interval) if base_pitch is not None else None
                return chosen_interval, prob, top_candidates, next_pitch
            else:
                return 0, 1.0, [(0, 1.0)], (base_pitch if base_pitch is not None else None)
        keys = list(marg_counts.keys())
        counts = [marg_counts[k] for k in keys]
        chosen_key, chosen_prob, top_probs = pick_from_counts(keys, counts, n_candidates)
        if chosen_key is None:
            chosen_key = random.choice(keys)
            chosen_prob = 1.0 / len(keys)
            top_probs = [(k, chosen_prob) for k in keys[:min(4, len(keys))]]
        chosen_interval = int(chosen_key)
        next_pitch = (base_pitch + chosen_interval) if base_pitch is not None else None
        return chosen_interval, chosen_prob, top_probs, next_pitch

    # 5) on a dist = dict{interval: count} pour found_context
    keys = list(dist.keys())
    counts = [dist[k] for k in keys]

    # 6) On fait le tirage parmi top n_candidates (uniform sur les top)
    chosen_key, chosen_prob, top_probs = pick_from_counts(keys, counts, n_candidates)
    if chosen_key is None:
        # improbable, fallback marginal comme sécurité
        marg_counts = defaultdict(float)
        for sub_dist in inter_table.values():
            for iv, cnt in sub_dist.items():
                marg_counts[iv] += cnt
        keys = list(marg_counts.keys())
        counts = [marg_counts[k] for k in keys]
        chosen_key, chosen_prob, top_probs = pick_from_counts(keys, counts, n_candidates)
        if chosen_key is None:
            chosen_interval = 0
            chosen_prob = 1.0
            top_probs = [(0, 1.0)]
            next_pitch = (base_pitch + chosen_interval) if base_pitch is not None else None
            return chosen_interval, chosen_prob, top_probs, next_pitch

    chosen_interval = int(chosen_key)
    next_pitch = (base_pitch + chosen_interval) if base_pitch is not None else None
    return chosen_interval, chosen_prob, top_probs, next_pitch