from typing import Dict, List, Tuple, Any, Optional
import random
from tqdm import tqdm

class OracleBuilder:
    """
    Classe pour construire un oracle de facteurs à partir de symboles musicaux (notes et accords),
    en prenant en compte différents niveaux de similarité.
    """

    @staticmethod
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

    @staticmethod
    def key_similarity_level(key1: Tuple, key2: Tuple) -> Optional[int]:
        """
        Détermine le niveau de similarité entre deux clés de symbole musical.
        Retourne :
            3 : Similarité Forte (pitch/chord, duration, velocity identiques)
            2 : Similarité Moyenne (pitch/chord & duration identiques)
            1 : Similarité Faible (pitch/chord identiques)
            None : Pas de similarité
        """
        if key1[0] != key2[0]:
            return None
        # note: ("note", pitch, duration, velocity)
        # chord: ("chord", tuple(pitch), duration, velocity)
        # compare type-specific attributes
        # strong: full equality
        if key1 == key2:
            return 3
        # medium: type and pitch & duration
        if key1[1] == key2[1] and key1[2] == key2[2]:
            return 2
        # weak: type and pitch
        if key1[1] == key2[1]:
            return 1
        return None

    @staticmethod
    def build_oracle(sequence: List[Dict[str, Any]]) -> Tuple[
        Dict[int, Dict[Tuple, int]], Dict[int, int],
        Dict[int, Dict[Tuple, int]], Dict[int, int],
        Dict[int, Dict[Tuple, int]], Dict[int, int]
    ]:
        """
        Construit l’oracle à partir d’une séquence de symboles.

        Args:
            sequence: Liste de symboles (dictionnaires "note" ou "chord")

        Returns:
            Tuple contenant les transitions et supply pour les niveaux de similarité 3, 2 et 1.
        """
        # Initialisation des oracles pour chaque niveau
        levels = [3, 2, 1]
        transitions = {lvl: {0: {}} for lvl in levels}
        supply = {lvl: {0: -1} for lvl in levels}
        current_state = 0

        for symbol in tqdm(sequence, desc="Building Oracle"):
            sigma_key = OracleBuilder.symbol_to_key(symbol)
            new_state = current_state + 1

            # préparer nouveaux dictionnaires pour le nouvel état
            for lvl in levels:
                transitions[lvl][new_state] = {}

            # ajouter transitions et mettre à jour supply pour chaque niveau
            for lvl in levels:
                # ajout direct à partir de l'état courant
                transitions[lvl][current_state].setdefault(sigma_key, new_state)

                # mise à jour des suffixes
                k = supply[lvl][current_state]
                while k > -1 and OracleBuilder.key_similarity_level(sigma_key, sigma_key) is not None \
                      and all(
                          OracleBuilder.key_similarity_level(prev_key, sigma_key) != lvl
                          for prev_key in transitions[lvl][k]
                      ):
                    transitions[lvl][k][sigma_key] = new_state
                    k = supply[lvl][k]

                # détermination du supply pour le nouvel état
                if k == -1:
                    s = 0
                else:
                    # trouver un précédent de même similarité
                    s = next(
                        transitions[lvl][k][prev_key]
                        for prev_key in transitions[lvl][k]
                        if OracleBuilder.key_similarity_level(prev_key, sigma_key) == lvl
                    )
                supply[lvl][new_state] = s

            current_state = new_state

        # retourner dans l'ordre (3, supply3, 2, supply2, 1, supply1)
        return (
            transitions[3], supply[3],
            transitions[2], supply[2],
            transitions[1], supply[1]
        )


def generate_note_oracle(
    previous_state: int,
    transitions: Dict[int, Dict[Tuple[Any, ...], int]],
    supply: Dict[int, int],
    symbols: List[Dict],
    target_duration: int,
    gap: int,
    p: float = 0.8,
    contour: bool = True
) -> Tuple[int, List[int], str]:
    """
    Generate the next note or chord using one level of the factor oracle.

    Args:
        previous_state: last state index in the oracle.
        transitions: mapping state -> {symbol_key: next_state}.
        supply: mapping state -> suffix_state.
        symbols: list of original symbols (with 'type', 'pitch'/'pitch', 'duration', 'velocity', 'onset').
        target_duration: duration target for filtering.
        gap: melodic contour direction (+ up, - down, 0 neutral).
        p: probability of following a factor link.
        contour: whether to enforce melodic contour.

    Returns:
        next_state: chosen state index.
        new_symbol: List pitch for a note or tuple of pitch for a chord.
        link_type: which link was used ('factor', 'suffix', or 'fallback').
    """
    # Build arrays aligned to oracle states (state 0 dummy)
    rep_pitch: List[int] = [0]
    durations: List[int] = [0]
    originals: List[List[int]] = [[]]  # now holds lists, index 0 unused

    for sym in symbols:
        dur = sym['duration']
        if sym['type'] == 'note':
            pr = int(sym['pitch'])
            orig = [pr]
        elif sym['type'] == 'chord':
            orig = list(sym['pitch'])
            pr = orig[0]
        else:
            raise ValueError(f"Unknown symbol type: {sym.get('type')}")
        originals.append(orig)
        rep_pitch.append(pr)
        durations.append(dur)

    max_state = len(symbols)
    next_state: Optional[int] = None
    link_type = 'fallback'

    # factor / suffix links logic
    state_links = transitions.get(previous_state, {})
    if state_links:
        if random.random() < p:
            link_type = 'factor'
            candidates = list(state_links.values())
            if contour and previous_state > 0:
                cur = rep_pitch[previous_state]
                filtered = [
                    s for s in candidates
                    if 0 < s <= max_state and (
                        (gap > 0 and rep_pitch[s] > cur) or
                        (gap < 0 and rep_pitch[s] < cur) or
                        (gap == 0)
                    )
                ]
                next_state = (min(filtered, key=lambda s: abs(durations[s] - target_duration))
                              if filtered else random.choice(candidates))
            else:
                next_state = random.choice(candidates)
        else:
            link_type = 'suffix'
            sl = supply.get(previous_state, -1)
            if sl != -1:
                cand = sl + 1
                next_state = cand if 0 < cand <= max_state else 0
            else:
                next_state = random.choice(list(state_links.values())) if state_links else previous_state

    # fallback if nothing picked or reached the last state
    if next_state is None or not (0 < next_state <= max_state):
        sl = supply.get(previous_state, -1)
        if previous_state == max_state or sl == -1:
            sl = supply.get(max_state, -1)  # Follow suffix from the last state
        if sl != -1 and (sl + 1) <= max_state:
            next_state = sl + 1
            link_type = 'suffix'
        else:
            next_state = 1  # Ultimate fallback
            link_type = 'wrap-around'

    new_symbol: List[int] = originals[next_state]
    return next_state, new_symbol, link_type