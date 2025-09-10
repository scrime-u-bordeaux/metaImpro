from typing import Dict, List, Optional, Tuple, Any
import random
from tqdm import tqdm

class OracleBuilder:
    @staticmethod
    def symbol_to_key(symbol: Dict[str, Any]) -> str:
        """
        Transforme un symbole (note ou accord dict) en chaîne de caractères hashable.
        """
        if symbol["type"] == "note":
            return f"note-{symbol['pitch']}-{symbol['duration']}-{symbol['velocity']}"
        elif symbol["type"] == "chord":
            sorted_pitches = sorted(symbol["pitch"])
            pitches_str = ','.join(map(str, sorted_pitches))
            return f"chord-{pitches_str}-{symbol['duration']}-{symbol['velocity']}"
        else:
            raise ValueError(f"Type de symbole inconnu: {symbol.get('type')}")

    @staticmethod
    def key_similarity_level(key1: str, key2: str) -> Optional[int]:
        """
        Détermine le niveau de similarité entre deux clés de symbole musical.
        """
        def parse_key(key):
            parts = key.split('-')
            type_ = parts[0]
            if type_ == "note":
                pitch, duration, velocity = map(int, parts[1:])
                return type_, pitch, duration, velocity
            elif type_ == "chord":
                pitches = list(map(int, parts[1].split(',')))
                duration, velocity = map(int, parts[2:4])
                return type_, list(pitches), duration, velocity
            else:
                raise ValueError(f"Type de clé inconnu: {type_}")

        type1, pitch1, dur1, vel1 = parse_key(key1)
        type2, pitch2, dur2, vel2 = parse_key(key2)

        if type1 != type2:
            return None

        if pitch1 == pitch2 and dur1 == dur2 and vel1 == vel2:
            return 3
        if pitch1 == pitch2 and dur1 == dur2:
            return 2
        if pitch1 == pitch2:
            return 1
        return None

    @staticmethod
    def build_oracle(sequence: List[Dict[str, Any]]) -> Tuple[
        Dict[int, Dict[str, int]], Dict[int, int],
        Dict[int, Dict[str, int]], Dict[int, int],
        Dict[int, Dict[str, int]], Dict[int, int]
    ]:
        levels = [3, 2, 1]
        transitions = {lvl: {0: {}} for lvl in levels}
        supply = {lvl: {0: -1} for lvl in levels}
        current_state = 0

        for symbol in tqdm(sequence, desc="Building Oracle"):
            sigma_key = OracleBuilder.symbol_to_key(symbol)
            new_state = current_state + 1

            for lvl in levels:
                transitions[lvl][new_state] = {}

            for lvl in levels:
                transitions[lvl][current_state].setdefault(sigma_key, new_state)
                k = supply[lvl][current_state]

                while k > -1 and OracleBuilder.key_similarity_level(sigma_key, sigma_key) is not None \
                        and all(
                            OracleBuilder.key_similarity_level(prev_key, sigma_key) != lvl
                            for prev_key in transitions[lvl][k]
                        ):
                    transitions[lvl][k][sigma_key] = new_state
                    k = supply[lvl][k]

                if k == -1:
                    s = 0
                else:
                    s = next(
                        transitions[lvl][k][prev_key]
                        for prev_key in transitions[lvl][k]
                        if OracleBuilder.key_similarity_level(prev_key, sigma_key) == lvl
                    )
                supply[lvl][new_state] = s

            current_state = new_state

        return (
            transitions[3], supply[3],
            transitions[2], supply[2],
            transitions[1], supply[1]
        )

def generate_note_oracle(
    previous_state: int,
    transitions: Dict[int, Dict[str, int]],
    supply: Dict[int, int],
    symbols: List[Dict],
    target_duration: int,
    gap: int,
    p: float = 0.8,
    contour: bool = True
) -> Tuple[int, List[int], str]:
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

    if next_state is None or not (0 < next_state <= max_state):
        sl = supply.get(previous_state, -1)
        if previous_state == max_state or sl == -1:
            sl = supply.get(max_state, -1)
        if sl != -1 and (sl + 1) <= max_state:
            next_state = sl + 1
            link_type = 'suffix'
        else:
            next_state = 1
            link_type = 'wrap-around'

    new_symbol: List[int] = originals[next_state]
    return next_state, new_symbol, link_type
