import factor_oracle as fo
import random
import numpy as np
"""
Ce code contient des fonctions pour générer des séquences de notes
"""


def generate_sequence_simple_oracle(transitions, supply, p=0.8, steps=100):
    """
    Génère une séquence aléatoire en suivant les transitions du Factor Oracle.
    - transitions: dict des transitions principales
    - supply: dict des suffix links
    - p: probabilité de suivre un factor link
    - steps: nombre d'étapes à générer
    """
    sequence = []
    state = 0
    max_state = max(transitions.keys())  # Or: len(midSymbols) - 1

    for _ in range(steps):
        next_state = None

        if state in transitions and transitions[state]:
            if random.random() < p:
                next_state = random.choice(list(transitions[state].values()))
            elif state in supply and supply[state] != -1:
                next_state = supply[state]

        # If no valid transition or suffix, move forward *only* if within range
        if next_state is None:
            if state + 1 <= max_state:
                next_state = state + 1
            else:
                # Restart from 0 or pick a random valid state
                next_state = 0  # or: random.randint(0, max_state)

        sequence.append(next_state)
        state = next_state

    return sequence

def generate_sequence_improved_oracle(transitions, supply, p=0.8, steps=100):
    """
    Génère une séquence aléatoire en suivant les transitions du Factor Oracle.
    - transitions: dict des transitions principales
    - supply: dict des suffix links
    - p: probabilité de suivre un factor link
    - steps: nombre d'étapes à générer
    """
    sequence = []
    state = 0
    max_state = max(transitions.keys())  # Or: len(midSymbols) - 1

    for _ in range(steps):
        next_state = None

        # Decide whether to follow a factor link or a suffix link
        if state in transitions and transitions[state]:
            if random.random() < p:
                # Follow a factor link
                next_state = random.choice(list(transitions[state].values()))
            elif state in supply and supply[state] != -1:
                # Follow a suffix link
                next_state = supply[state]
                # Immediately move to the next state after a suffix jump
                if next_state + 1 <= max_state:
                    sequence.append(next_state)
                    next_state += 1
                else:
                    # Restart from 0 if out of bounds
                    next_state = 0

        # If no valid transition or suffix, move forward *only* if within range
        if next_state is None:
            if state + 1 <= max_state:
                next_state = state + 1
            else:
                # Restart from 0 or pick a random valid state
                next_state = 0  # or: random.randint(0, max_state)

        sequence.append(next_state)
        state = next_state

    return sequence


def generate_note_oracle(previous_state, duration, transitions, supply, midSymbols, gap, p=0.8):
    """
    Génère un nouvel état (indice dans l'oracle) et retourne la note associée,
    qui est un tuple (pitch, duration, velocity).
    """
    max_state = max(transitions.keys()) if transitions else 0
    next_state = None

    if previous_state in transitions and transitions[previous_state]:
        # Branche principale (probabilité p) : explorer via factor links
        if random.random() < p:
            candidates = list(transitions[previous_state].values())

            # Filtrer selon le contour (gap)
            filtered = []
            curr_pitch = midSymbols[previous_state][0]
            for s in candidates:
                if s < len(midSymbols):
                    next_pitch = midSymbols[s][0]
                    if (gap > 0 and next_pitch > curr_pitch) or \
                       (gap < 0 and next_pitch < curr_pitch) or \
                       (gap == 0):
                        filtered.append(s)

            if filtered:
                # choisir la durée la plus proche
                next_state = min(filtered, key=lambda s: abs(midSymbols[s][1] - duration))
            else:
                # pas de candidat respectant le gap : fallback stochastique parmi tous les factor links
                next_state = random.choice(candidates)
        else:
            # Sinon, branche suffix link (supply) puis +1 pour éviter la redondance
            sl = supply.get(previous_state, None)
            if sl is not None and sl != -1:
                # on ajoute +1 et on wrappe
                if sl + 1 <= max_state:
                    next_state = sl + 1
                else:
                    next_state = 0
            else:
                # si pas de suffix link valide, retomber sur les factor links
                next_state = random.choice(list(transitions[previous_state].values()))

    # S'assurer d'un état valide
    if next_state is None or not (0 <= next_state < len(midSymbols)):
        # on reste sur le même état si tout échoue
        next_state = previous_state if 0 <= previous_state < len(midSymbols) else 0

    # Construire et renvoyer la note
    base = midSymbols[next_state]
    new_note = (base[0], duration, base[2])
    return next_state, new_note



def generate_note_markov(previous_pitch, transition_matrix: np.ndarray, notes, gap) -> int:
    """
    Génère le pitch suivant selon une chaîne de Markov construite sur les pitches,
    représentée par une matrice de transition et une collection de notes.

    Args:
        previous_pitch (int): pitch de la note précédente.
        transition_matrix (np.ndarray): matrice de transition calculé à l'aide de transition_matrix
        notes (list[int] or np.ndarray): collection des pitches correspondant aux indices.
        gap (int): écart entre les touches appuyées (positif pour monter, négatif pour descendre, zéro pour neutre)

    Returns:
        int: le next_pitch généré.
    """
    # o*On transforme notes en array numpy
    notes_arr = np.array(notes)

    # On trouve l’indice du pitch précédent
    idxs = np.where(notes_arr == previous_pitch)[0]
    if len(idxs) > 0:
        idx = int(idxs[0])
    else:
        idx = 0  # fallback si le pitch n'est pas dans notes

    row = transition_matrix[idx]

    if row.sum() > 0:
        # Filtre selon le gap
        if gap > 0:
            mask = notes_arr > previous_pitch
        elif gap < 0:
            mask = notes_arr < previous_pitch
        else:
            mask = np.ones_like(row, dtype=bool)

        # Applique le masque sur la distribution
        filtered = row * mask
        if filtered.sum() > 0:
            probs = filtered / filtered.sum()
            next_idx = np.random.choice(len(row), p=probs)
        else:
            # Si aucun élément validé par le masque, retombe sur la distribution totale
            probs = row / row.sum()
            next_idx = np.random.choice(len(row), p=probs)
    else:
        # fallback uniforme si la ligne est nulle
        next_idx = np.random.choice(len(row))

    return int(notes_arr[next_idx])

