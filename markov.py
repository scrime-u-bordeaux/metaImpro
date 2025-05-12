import numpy as np
from collections import defaultdict

def build_vlmc_table(midSymbols, max_order=3):
    """
    Construit une table de transition pour une chaîne de Markov à longueur variable.

    Args:
        midSymbols (list of tuples): Séquence de notes (pitches, duration, velocity).
        max_order (int): Ordre maximal du contexte (par défaut 3).

    Returns:
        dict: Table de transition sous forme de dictionnaire.
    """
    sequence = [s[0] for s in midSymbols]
    table = defaultdict(lambda: defaultdict(int))
    for i in range(len(sequence)):
        for order in range(1, max_order + 1):
            if i - order < 0:
                continue
            context = tuple(sequence[i - order:i])
            next_note = sequence[i]
            table[context][next_note] += 1
    return table

def generate_note_vlmc(previous_notes, vlmc_table, notes, gap, contour=True, max_order=3):
    """
    Génère la note suivante en utilisant une chaîne de Markov à longueur variable à  partir de la note précédente.

    Args:
        previous_notes (list of int): Notes précédentes (historique).
        vlmc_table (dict): Table de transition VLMC.
        notes (list of int): Liste des notes possibles.
        gap (int): Écart entre les touches appuyées (positif pour monter, négatif pour descendre, zéro pour neutre).
        contour (bool): Active ou non le contour mélodique.
        max_order (int): Ordre maximal du contexte.

    Returns:
        tuple: (next_pitch, next_prob, top_probs)
    """
    notes_arr = np.array(notes)
    context = tuple(previous_notes[-max_order:]) if previous_notes else ()
    
    # Recherche du contexte le plus long possible
    for order in range(len(context), 0, -1):
        sub_context = context[-order:]
        if sub_context in vlmc_table:
            next_notes_dict = vlmc_table[sub_context]
            break
    else:
        # Si aucun contexte trouvé, sélection aléatoire uniforme
        next_pitch = int(np.random.choice(notes_arr))
        return next_pitch, 1.0 / len(notes_arr), [(next_pitch, 1.0 / len(notes_arr))]

    # Construction de la distribution de probabilité
    next_notes = list(next_notes_dict.keys())
    counts = np.array([next_notes_dict[n] for n in next_notes])
    probs = counts / counts.sum()

    # Application du contour mélodique si activé
    if contour and previous_notes:
        last_pitch = previous_notes[-1]
        if gap > 0:
            mask = np.array(next_notes) > last_pitch
        elif gap < 0:
            mask = np.array(next_notes) < last_pitch
        else:
            mask = np.ones_like(probs, dtype=bool)
        if mask.any():
            probs = probs[mask]
            probs = probs / probs.sum() #renormalisation
            next_notes = np.array(next_notes)[mask]
        else:
            # Si aucune note ne respecte le contour, fallback uniforme
            next_pitch = int(np.random.choice(notes_arr))
            return next_pitch, 1.0 / len(notes_arr), [(next_pitch, 1.0 / len(notes_arr))]

    # Sélection de la note suivante
    next_pitch = int(np.random.choice(next_notes, p=probs))
    next_prob = float(probs[np.where(next_notes == next_pitch)[0][0]])

    # Identification des 2 notes les plus probables
    top_indices = np.argsort(probs)[::-1][:2]
    top_probs = [(int(next_notes[i]), float(probs[i])) for i in top_indices]

    return next_pitch, next_prob, top_probs