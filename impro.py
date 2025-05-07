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


def generate_note_oracle(previous_state, duration, transitions, supply, midSymbols, gap, p=0.8, contour = True):
    """
    Args:
        previous_state (int): état de la note précédente dans le dictionnaire des transitions.
        duration (int): Durée en seconde de la note précédnte et du silence entre la note précédente et la note suivante
        transitions (dict[int, dict]): dictionnaire des transitions de l'oracle
        supply (dict[int, int]) : dictionnaire de la supply function de l'oracle
        midSymbols (liste[tuple])) : liste des (pictchs, duration, velocity)
        gap (int): écart entre les touches appuyées (positif pour monter, négatif pour descendre, zéro pour neutre)
        Contour (bool): Active où non le contour mélodique
    Returns:
        int: le next_pitch généré.
    """
    max_state = max(transitions.keys()) if transitions else 0
    next_state = None
    links = None
    if previous_state in transitions and transitions[previous_state]:
        # Branche principale (probabilité p) : explorer via factor links
        if random.random() < p:
            links= 'factor'
            candidates = list(transitions[previous_state].values())

            if contour:
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
                    # Choisir la durée la plus proche
                    next_state = min(filtered, key=lambda s: abs(midSymbols[s][1] - duration))
                else:
                    # Aucun filtré => choix aléatoire
                    next_state = random.choice(candidates)
            else:
                # pas de candidat respectant le gap : fallback stochastique parmi tous les factor links
                next_state = random.choice(candidates)
        else:
            links = 'suffix'
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
        links = 'fallback'
        # on reste sur le même état si tout échoue
        next_state = previous_state if 0 <= previous_state < len(midSymbols) else 0

    # Construire et renvoyer la note
    base = midSymbols[next_state]
    new_note = (base[0], duration, base[2])
    return next_state, new_note, links



def generate_note_markov(previous_pitch, transition_matrix: np.ndarray, notes, gap, contour = True) -> int:
    """
    Génère le pitch suivant selon une chaîne de Markov construite sur les pitches,
    représentée par une matrice de transition et une collection de notes.

    Args:
        previous_pitch (int): pitch de la note précédente.
        transition_matrix (np.ndarray): matrice de transition calculé à l'aide de transition_matrix
        notes (list[int] or np.ndarray): collection des pitches correspondant aux indices.
        gap (int): écart entre les touches appuyées (positif pour monter, négatif pour descendre, zéro pour neutre)
        Contour (bool): Active où non le contour mélodique
    Returns:
        next_pitch (int): le pitch généré.
        next_prob (float): probabilité associée à ce pitch dans la distribution utilisée.
        top_probs (list of tuples): les deux notes les plus probables sous forme [(note1, prob1), (note2, prob2)].
    """
    # On transforme notes en array numpy
    notes_arr = np.array(notes)

    # On trouve l’indice du pitch précédent
    idxs = np.where(notes_arr == previous_pitch)[0]
    idx = int(idxs[0]) if len(idxs) > 0 else 0

    row = transition_matrix[idx]

    if row.sum() > 0:
        # Détermination du masque selon contour
        if contour:
            if gap > 0:
                mask = notes_arr > previous_pitch
            elif gap < 0:
                mask = notes_arr < previous_pitch
            else:
                mask = np.ones_like(row, dtype=bool)
        else:
            # Sans contour => tous valides
            mask = np.ones_like(row, dtype=bool)

        # Applique le masque sur la distribution
        filtered = row * mask
        if filtered.sum() > 0:
            probs = filtered / filtered.sum()
        else:
            # fallback uniforme si pas de transitions
            probs = np.ones_like(row) / row.size

    # Sélectionne un index selon la distribution
    next_idx = np.random.choice(probs.size, p=probs)

    # Détermine les valeurs de retour
    next_pitch = int(notes_arr[next_idx])
    next_prob = float(probs[next_idx])

    # Identifie les 2 meilleures probabilités
    sorted_idx = np.argsort(row)[::-1]
    top_idx = sorted_idx[:4] if sorted_idx.size >= 4 else sorted_idx
    total = row.sum() if row.sum() > 0 else 1.0
    top_probs = [(int(notes_arr[i]), float(row[i] / total)) for i in top_idx]

    return next_pitch, next_prob, top_probs

