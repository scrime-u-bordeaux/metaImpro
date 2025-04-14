import factor_oracle as fo
import random

"""
Ce code contient des fonctions pour générer des séquences de notes
"""


def generate_sequence_simple(transitions, supply, p=0.8, steps=100):
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

def generate_sequence_improved(transitions, supply, p=0.8, steps=100):
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

def generate_note(previous_state, duration, transitions, supply, midSymbols, p=0.8):
    """
    Génère un nouvel état (indice dans l'oracle) et retourne la note associée, qui est un tuple (pitch, duration, velocity).
    La durée passée est insérée dans la note.
    
    Args:
        previous_state (int): état courant (indice dans l'oracle).
        duration (float): durée d'appui de la touche (en secondes).
        transitions (dict): dictionnaire des transitions du Factor Oracle.
        supply (dict): dictionnaire des suffix links.
        midSymbols (list): liste des symboles (tuples de type (pitch, duration, velocity)) initialement extraits du fichier MIDI.
        p (float): probabilité de suivre une factor link.
        
    Returns:
        (new_state, note): new_state est l'indice de l'état généré, note est le tuple (pitch, duration, velocity).
    """
    next_state = None
    max_state = max(transitions.keys())
    
    # On vérifie que l'état courant possède des transitions
    if previous_state in transitions and transitions[previous_state]:
        if random.random() < p:
            # Suivre une transition principale (factor link)
            next_state = random.choice(list(transitions[previous_state].values()))
        elif previous_state in supply and supply[previous_state] != -1:
            # Suivre la suffix link
            next_state = supply[previous_state]
            if next_state + 1 <= max_state:
                next_state += 1
            else:
                next_state = 0
    # Si aucune transition n'est disponible ou n'a été appliquée
    if next_state is None:
        if previous_state + 1 <= max_state:
            next_state = previous_state + 1
        else:
            next_state = 0

    # Récupérer la note de base correspondant au nouvel état.
    # Attention : dans l'oracle, l'état 0 est souvent un état initial fictif.
    try:
        base_symbol = midSymbols[next_state]
    except IndexError:
        # Si par hasard l'indice est hors gamme, on fournit une valeur par défaut.
        base_symbol = (60, 0.1, 64)
    
    # On insère la durée mesurée dans la note générée
    new_note = (base_symbol[0], duration, base_symbol[2])
    return next_state, new_note