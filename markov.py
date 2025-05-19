from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Any


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


def build_vlmc_table(
    symbols: List[Dict[str, Any]], max_order: int = 3
    ) -> Dict[Tuple, Dict[Tuple, int]]:
    """
    Construit la table VLMC à partir d'une séquence de symboles.
    """
    table: Dict[Tuple, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))
    keys = [symbol_to_key(s) for s in symbols]

    for i in range(len(keys)):
        for order in range(1, max_order + 1):
            if i - order < 0:
                break
            context = tuple(keys[i - order : i])
            table[context][keys[i]] += 1

    return table



def generate_symbol_vlmc(
    previous_symbols: List[Dict[str, Any]],
    vlmc_table: Dict[Tuple, Dict[Tuple, int]],
    all_keys: List[Tuple],
    max_order: int = 3,
    gap: int = 0,
    contour: bool = True
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

    Returns:
        sym: nouveau symbole dict.
        prob: probabilité associée.
        top_probs: liste des 4 meilleures probabilités [(sym, prob), ...].
    """
    # Construction de l'historique de clés
    history = [symbol_to_key(s) for s in previous_symbols]
    context = tuple(history[-max_order:]) if history else ()

    # Recherche du plus long contexte disponible
    for order in range(len(context), 0, -1):
        sub = context[-order:]
        if sub in vlmc_table:
            dist = vlmc_table[sub]
            break
    else:
        # fallback uniforme sur toutes les clés
        idx = np.random.choice(len(all_keys))
        key = all_keys[idx]
        sym = key_to_symbol(key)
        prob = 1.0 / len(all_keys)
        return sym, prob, [(sym, prob)]

    # Préparation de la distribution
    symbols = list(dist.keys())
    counts = np.array([dist[k] for k in symbols], dtype=float)
    # Application du contour mélodique si activé
    if contour and previous_symbols:
        prev_pitch = previous_symbols[-1].get('pitch', 0)
        # Calcul des écarts
        pitches = np.array([key_to_symbol(k).get('pitch', 0) for k in symbols])
        if gap > 0:
            mask = pitches > prev_pitch
        elif gap < 0:
            mask = pitches < prev_pitch
        else:
            mask = np.ones_like(counts, dtype=bool)
        # Filtrer les counts
        filtered = counts * mask
        if filtered.sum() > 0:
            counts = filtered
    
    probs = counts / counts.sum()

    # Tirage
    idx = np.random.choice(len(symbols), p=probs)
    chosen_key = symbols[idx]
    sym = key_to_symbol(chosen_key)
    prob = float(probs[idx])

    # Sélection des 4 meilleurs
    top_n = min(4, len(symbols))
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_probs = [(key_to_symbol(symbols[i]), float(probs[i])) for i in top_idx]

    return sym, prob, top_probs



"""   
Utilisation: 
processor = MidiSymbolProcessor()
symbols = processor.process_midi_file("ton_fichier.mid")

vlmc_table = build_vlmc_table(symbols, max_order=3)
all_keys = list({symbol_to_key(s) for s in symbols})  # liste unique des clés symboles

previous = symbols[:3]  # par exemple, contexte initial
next_sym, prob, top = generate_symbol_vlmc(previous, vlmc_table, all_keys)
print(next_sym, prob, top)
"""