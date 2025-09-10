from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Any
import random
from music21 import pitch, scale
import re

def symbol_to_key(symbol: Any) -> Tuple:
    """
    Transforme un symbole (note ou accord) ou un pitch simple en tuple hashable.
    Accepte:
      - dict comme avant
      - int    -> note avec durée=1.0, velocity=100
      - list/tuple d'int -> accord avec durée=1.0, velocity=100
    """
    # Simple note pitch
    if isinstance(symbol, int):
        return ("note", symbol, 1.0, 100)

    # Simple chord as list/tuple of pitches
    if isinstance(symbol, (list, tuple)) and all(isinstance(p, int) for p in symbol):
        return ("chord", tuple(sorted(symbol)), 1.0, 100)

    # Existing dict handling
    if isinstance(symbol, dict):
        if symbol.get("type") == "note":
            return ("note", symbol["pitch"], symbol.get("duration", 1.0), symbol.get("velocity", 100))
        elif symbol.get("type") == "chord":
            return (
                "chord",
                tuple(sorted(symbol["pitch"])),
                symbol.get("duration", 1.0),
                symbol.get("velocity", 100),
            )
    
    raise ValueError(f"Type de symbole inconnu ou non supporté: {symbol!r}")



def key_to_symbol(key: Tuple) -> Dict[str, Any]:
    """
    Convertit une clé tuple en symbole dict.
    (inchangé)
    """
    if not key:
        raise ValueError("Clé vide fournie à key_to_symbol.")

    t = key[0]
    if t == "note":
        if len(key) == 4:
            _, pitch, duration, velocity = key
        elif len(key) == 3:
            _, pitch, duration = key
            velocity = 100
        elif len(key) == 2:
            _, pitch = key
            duration, velocity = 1.0, 100
        else:
            raise ValueError(f"Clé 'note' invalide : {key}")
        return {"type": "note", "pitch": pitch, "duration": duration, "velocity": velocity}

    elif t == "chord":
        if len(key) == 4:
            _, pitches, duration, velocity = key
        elif len(key) == 3:
            _, pitches, duration = key
            velocity = 100
        elif len(key) == 2:
            _, pitches = key
            duration, velocity = 1.0, 100
        else:
            raise ValueError(f"Clé 'chord' invalide : {key}")
        return {"type": "chord", "pitch": list(pitches), "duration": duration, "velocity": velocity}

    else:
        raise ValueError(f"Type de clé inconnu : {t}")


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
    symbols: List[Any], max_order: int = 3, similarity_level: int = 3
) -> Dict[Tuple, Dict[Tuple, int]]:
    """
    Construit la table VLMC: contexte → {successeur → count}.
    Accepte symbols sous forme de dict, int, ou list[int].
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

CHORD_TO_SCALE = {
    # Accords Majeurs et leurs extensions
    'maj13': scale.LydianScale,
    'maj11': scale.LydianScale,
    'maj9': scale.LydianScale,
    'maj7': scale.MajorScale,
    '6': scale.MajorScale,
    # Accords mineurs
    'mmaj7': scale.MelodicMinorScale,
    'm7': scale.DorianScale,
    'min7': scale.DorianScale,
    'm6': scale.DorianScale,
    'm': scale.HypoaeolianScale,
    # Dominantes
    '7b13': scale.MixolydianScale,
    '7': scale.MixolydianScale,
    # Diminués et demi‑diminués
    'dim7': scale.OctatonicScale,
    'ø7': scale.LocrianScale,
    'm7b5': scale.LocrianScale,
    # Sus2 / Sus4
    'sus2': scale.LydianScale,
    'sus4': scale.LydianScale,
    # Hexatoniques, whole‑tone…
    'aug': scale.WholeToneScale,
    '+': scale.WholeToneScale,
}

def is_white_note(index):
    """
    Args:
        index (int): Index de la note MIDI (0-127)
    Returns:
        bool
    """
    WHITE_SEMITONES = {0, 2, 4, 5, 7, 9, 11}
    return  (index % 12) in WHITE_SEMITONES

def filter_by_scale(symbols_list, counts, current_chord, strict_mode=False):
    """
    Filtre les symboles candidats selon la gamme de l'accord courant
    
    Args:
        symbols_list: Liste des clés (tuples) candidates du VLMC
        counts: Array numpy des comptages correspondants
        current_chord: Accord courant (ex: "C7", "Fm7")
        strict_mode: Si True, rejette les accords avec des notes hors gamme
                    Si False, accepte les accords avec au moins une note dans la gamme
    Returns:
        (filtered_symbols, filtered_counts): Listes filtrées
    """
    if not current_chord:
        return symbols_list, counts
    
    # Filtrer les None dès le début
    valid_pairs = [(symbol, count) for symbol, count in zip(symbols_list, counts) if symbol is not None]
    if not valid_pairs:
        return symbols_list, counts
    
    symbols_list, counts = zip(*valid_pairs)
    symbols_list, counts = list(symbols_list), np.array(counts, dtype=float)  # Ensure counts is numpy array
    
    try:
        # Parser l'accord pour extraire root et type
        CHORD_ROOT_RE = re.compile(r'^([A-Ga-g][#b]?)(.*)$')
        figure = current_chord.split('/')[0].strip()
        m = CHORD_ROOT_RE.match(figure)
        if not m:
            return symbols_list, counts
            
        root, chord_type = m.group(1), m.group(2).strip()
        
        # Récupérer la classe de gamme
        scale_class = CHORD_TO_SCALE.get(chord_type)
        if not scale_class:
            # Type d'accord inconnu, on ne filtre pas
            return symbols_list, counts
        
        # Créer la gamme
        root_pitch = pitch.Pitch(root)
        chord_scale = scale_class(root_pitch)
        
        # Générer les classes de hauteur de la gamme (0-11)
        scale_pitch_classes = set()
        for degree in range(1, 8):
            try:
                scale_pitch = chord_scale.pitchFromDegree(degree)
                scale_pitch_classes.add(scale_pitch.midi % 12)
            except:
                continue
        
        # Filtrer les symboles
        filtered_symbols = []
        filtered_counts = []
        
        for symbol_key, count in zip(symbols_list, counts):
            # Vérification supplémentaire au cas où un None aurait échappé
            if symbol_key is None:
                continue
                
            keep_symbol = False
            
            if symbol_key[0] == "note":
                # Note simple
                pitch_class = symbol_key[1] % 12
                if pitch_class in scale_pitch_classes:
                    keep_symbol = True
            elif symbol_key[0] == "chord":
                # Accord
                pitches = symbol_key[1]
                in_scale_count = sum(1 for p in pitches if (p % 12) in scale_pitch_classes)
                
                if strict_mode:
                    # Toutes les notes doivent être dans la gamme
                    keep_symbol = (in_scale_count == len(pitches))
                else:
                    # Au moins une note doit être dans la gamme
                    keep_symbol = (in_scale_count > 0)
            
            if keep_symbol:
                filtered_symbols.append(symbol_key)
                filtered_counts.append(count)
        
        # Si aucun symbole ne passe le filtre, retourner les originaux
        if not filtered_symbols:
            return symbols_list, counts  # counts is already a numpy array now
            
        return filtered_symbols, np.array(filtered_counts, dtype=float)
        
    except Exception as e:
        print(f"Erreur dans filter_by_scale: {e}")
        return symbols_list, counts  # counts is already a numpy array now
    

def generate_symbol_vlmc(
    previous_symbols: List[Any],
    vlmc_table: Dict[Tuple, Dict[Tuple, int]],
    max_order: int = 3,
    gap: int = 0,
    contour: bool = True,
    similarity_level: int = 3,
    n_candidates: int = 1,
    current_chord = None,
    use_scale_filter: bool = True
) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
    """
    Génère le symbole suivant via VLMC avec logique de contour mélodique.
    La première note (fallback) est tirée selon la distribution marginale observée.

    Args:
        n_candidates: nombre de candidats à considérer pour le tirage.
                      1 -> prend toujours le max;
                      2 -> tirage aléatoire uniforme parmi les deux plus probables;
                      etc.
    """
    # 1) Préparer l’historique
    full_history = [symbol_to_key(s) for s in previous_symbols]
    trunc_history = [truncate_key(k, similarity_level) for k in full_history]
    context = tuple(trunc_history[-max_order:]) if trunc_history else ()

    # Fonction interne pour tirer selon top n_candidates
    def pick_from_probs(keys, probs):
        # indices triés par prob décroissante
        order_idx = np.argsort(probs)[::-1]
        top_k = min(n_candidates, len(keys))
        candidates_idx = order_idx[:top_k]
        # tirage uniforme parmi ces candidats
        chosen = np.random.choice(candidates_idx)
        return chosen

    # 2) Back‑off : chercher le plus long suffixe du contexte
    for order in range(len(context), 0, -1):
        sub = context[-order:]
        if sub in vlmc_table:
            dist = vlmc_table[sub]
            print(dist)
            # DEBUG
            print(f"[DEBUG] Context {sub!r} has {len(dist)} successors:")
            for k, c in dist.items():
                print("   ", k, "→", c)
            break
    else:
        # Fallback marginal
        print(f"[DEBUG-L{similarity_level}] Pas de successeurs  à  {context!r}")
        marg_counts: Dict[Tuple, float] = defaultdict(float)
        for sub_dist in vlmc_table.values():
            for key_, cnt in sub_dist.items():
                marg_counts[key_] += cnt
        keys = list(marg_counts.keys())
        counts = np.array([marg_counts[k] for k in keys], dtype=float)
        probs = counts / counts.sum()
        idx = pick_from_probs(keys, probs)
        chosen_key = keys[idx]
        sym = key_to_symbol(chosen_key)
        prob = float(probs[idx])
        top_idx = np.argsort(probs)[::-1][:min(4, len(keys))]
        top_probs = [(key_to_symbol(keys[i]), float(probs[i])) for i in top_idx]
        return sym, prob, top_probs

    # 3) Successeurs du contexte
    symbols_list = list(dist.keys())
    counts = np.array([dist[k] for k in symbols_list], dtype=float)

    # 4) Filtrage de contour
    if contour and previous_symbols:
        last_key = symbol_to_key(previous_symbols[-1])
        prev_pitch = last_key[1] if last_key[0] == "note" else last_key[1][0]
        flat = np.array([(k[1] if k[0] == "note" else k[1][0]) for k in symbols_list], dtype=float)
        mask = (flat > prev_pitch) if gap > 0 else (flat < prev_pitch) if gap < 0 else np.ones(len(symbols_list), bool)
        if mask.any():
            symbols_list = [s for s, m in zip(symbols_list, mask) if m]
            counts = np.array([dist[s] for s in symbols_list], dtype=float)

    # Filtrage des notes out
    if use_scale_filter and current_chord:
        symbols_list, counts = filter_by_scale(symbols_list, counts, current_chord, True)

    # 5) Fallback marginal si vide
    if len(symbols_list) == 0 or counts.sum() == 0:
        print(f"[DEBUG-L{similarity_level}] fallback marginal après filtrage pour contexte {context!r}")
        marg_counts: Dict[Tuple, float] = defaultdict(float)
        for sub_dist in vlmc_table.values():
            for key_, cnt in sub_dist.items():
                marg_counts[key_] += cnt
        keys = list(marg_counts.keys())
        counts = np.array([marg_counts[k] for k in keys], dtype=float)
        probs = counts / counts.sum()
        idx = pick_from_probs(keys, probs)
        chosen_key = keys[idx]
        sym = key_to_symbol(chosen_key)
        prob = float(probs[idx])
        top_idx = np.argsort(probs)[::-1][:min(4, len(keys))]
        top_probs = [(key_to_symbol(keys[i]), float(probs[i])) for i in top_idx]
        return sym, prob, top_probs

    # 6) Probabilités normales
    probs = counts / counts.sum()
    idx = pick_from_probs(symbols_list, probs)
    chosen_key = symbols_list[idx]
    sym = key_to_symbol(chosen_key)
    prob = float(probs[idx])
    top_idx = np.argsort(probs)[::-1][:min(4, len(symbols_list))]
    top_probs = [(key_to_symbol(symbols_list[i]), float(probs[i])) for i in top_idx]

    return sym, prob, top_probs

