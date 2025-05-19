from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Any
import random

def symbol_to_key(symbol: Dict[str, Any]) -> Tuple:
    """
    Transforme un symbole (note ou accord dict) en tuple hashable.
    """
    if symbol["type"] == "note":
        return ("note", symbol["pitch"], symbol["duration"], symbol["velocity"])
    elif symbol["type"] == "chord":
        return (
            "chord",
            tuple(sorted(symbol["pitch"])),
            symbol["duration"],
            symbol["velocity"],
        )
    else:
        raise ValueError(f"Type de symbole inconnu: {symbol.get('type')}")


def key_to_symbol(key: Tuple) -> Dict[str, Any]:
    """
    Convertit la clé tuple en symbole dict.
    """
    if key[0] == "note":
        return {
            "type": "note",
            "pitch": key[1],
            "duration": key[2],
            "velocity": key[3],
        }
    elif key[0] == "chord":
        return {
            "type": "chord",
            "pitch": list(key[1]),
            "duration": key[2],
            "velocity": key[3],
        }
    else:
        raise ValueError(f"Type de clé inconnue: {key[0]}")

def truncate_key(key: Tuple, similarity_level: int) -> Tuple:
    """
    Tronque une clé en fonction du niveau de similarité :
      3 → (type, pitch, duration, velocity)
      2 → (type, pitch, duration)
      1 → (type, pitch)
    """
    t, pitch, dur, vel = key
    if similarity_level == 3:
        return (t, pitch, dur, vel)
    elif similarity_level == 2:
        return (t, pitch, dur)
    elif similarity_level == 1:
        return (t, pitch)
    else:
        raise ValueError(f"Invalid similarity_level {similarity_level}")
    

def build_vlmc_table(
    symbols: List[Dict[str, Any]],
    max_order: int = 3,
    similarity_level: int = 3
) -> Dict[Tuple, Dict[Tuple, int]]:
    """
    Construit la table VLMC : contexte → {successeur → count}.
    """
    table: Dict[Tuple, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
    full_keys = [symbol_to_key(s) for s in symbols]

    for i in range(len(full_keys)):
        for order in range(1, max_order + 1):
            if i - order < 0:
                break
            # Contexte complet puis tronqué
            ctx_full = full_keys[i - order : i]
            ctx = tuple(truncate_key(k, similarity_level) for k in ctx_full)
            succ_full = full_keys[i]
            succ = truncate_key(succ_full, similarity_level)
            table[ctx][succ] += 1

    return table


def generate_symbol_vlmc(
    previous_symbols: List[Dict[str, Any]],
    vlmc_table: Dict[Tuple, Dict[Tuple, int]],
    all_keys: List[Tuple],
    max_order: int = 3,
    gap: int = 0,
    contour: bool = True, 
    similarity_level: int = 3
) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
    """
    Génère le symbole suivant via VLMC avec logique de contour mélodique.

    Args:
        previous_symbols: symboles précédents (notes/accords) sous forme de dictionnaires contenant au moins 'pitch'.
        vlmc_table: table VLMC (contexte -> {clé: count}).
        all_keys: toutes les clés possibles (fallback).
        max_order: ordre max du contexte.
        gap: écart entre les hauteurs (positif = monter, négatif = descendre).
        contour: active la contrainte de contour mélodique.
        similarity_level: niveau de similarité entre les symboles (3 ou 2 ou 1)

    Returns:
        sym: nouveau symbole dict.
        prob: probabilité associée.
        top_probs: liste des 4 meilleures probabilités [(sym, prob), ...].
    """
    # 1) Préparer l’historique  
    full_history = [symbol_to_key(s) for s in previous_symbols]
    trunc_history = [truncate_key(k, similarity_level) for k in full_history]
    context = tuple(trunc_history[-max_order:]) if trunc_history else ()

    # 2) Back‑off : chercher le plus long suffixe du contexte
    for order in range(len(context), 0, -1):
        sub = context[-order:]
        if sub in vlmc_table:
            dist = vlmc_table[sub]
            # DEBUG
            print(f"[DEBUG] Context {sub!r} has {len(dist)} successors:")
            for k, c in dist.items():
                print("   ", k, "→", c)
            break
    else:
        # Fallback uniforme
        key = random.choice(all_keys)
        sym = key_to_symbol(key)
        uni_p = 1.0 / len(all_keys)
        return sym, uni_p, [(sym, uni_p)]

    # 3) Extraire symboles et comptes
    symbols_list = list(dist.keys())
    counts = np.array([dist[k] for k in symbols_list], dtype=float)

    # 4) Filtrage de contour
    if contour and previous_symbols:
        last_key = symbol_to_key(previous_symbols[-1])
        prev_pitch = last_key[1] if last_key[0] == "note" else last_key[1][0]
        flat_pitches = np.array([
            (k[1] if k[0] == "note" else k[1][0])
            for k in symbols_list
        ], dtype=float)
        if gap > 0:
            mask = flat_pitches > prev_pitch
        elif gap < 0:
            mask = flat_pitches < prev_pitch
        else:
            mask = np.ones(len(symbols_list), dtype=bool)
        if mask.any():
            symbols_list = [s for (s, m) in zip(symbols_list, mask) if m]
            counts = np.array([dist[s] for s in symbols_list], dtype=float)

    # 5) Si rien à choisir → fallback uniforme
    if len(symbols_list) == 0 or counts.sum() == 0:
        key = random.choice(all_keys)
        sym = key_to_symbol(key)
        uni_p = 1.0 / len(all_keys)
        return sym, uni_p, [(sym, uni_p)]

    # 6) Calcul des probabilités
    probs = counts / counts.sum()

    # 7) Tirage et top‑4
    idx = np.random.choice(len(symbols_list), p=probs)
    chosen_key = symbols_list[idx]
    sym = key_to_symbol(chosen_key)
    prob = float(probs[idx])
    top_n = min(4, len(symbols_list))
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_probs = [
        (key_to_symbol(symbols_list[i]), float(probs[i]))
        for i in top_idx
    ]

    return sym, prob, top_probs