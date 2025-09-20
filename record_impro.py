import json
from datetime import datetime
import os

EVAL_DIR = "eval/"
def serialize_info(chord, pitch, onset, duration, velocity, effective_duration, is_black, desired, actual, success):
    """
    Normalise et retourne une entrée d'information pour historique/logging.
    - pitch : list/iterable de MIDI pitches
    - onset, duration, effective_duration : numeriques (ou None pour duration)
    - desired, actual : chaînes (p.ex. 'up'/'down'/'same') ou None
    - success : bool
    """
    # Normalisations défensives
    if pitch is None:
        pitch_list = []
    elif isinstance(pitch, (list, tuple)):
        pitch_list = [int(max(0, min(127, x))) for x in pitch]
    else:
        # cas où pitch est un entier unique
        pitch_list = [int(max(0, min(127, pitch)))]

    entry = {
        "current_chord": chord,
        "played_pitches": pitch_list,
        "onset": float(onset) if onset is not None else None,
        "duration": float(duration) if duration is not None else None,
        "velocity": int(velocity) if velocity is not None else None,
        "duration_effective": float(effective_duration) if effective_duration is not None else None,
        "is_black_key": bool(is_black),
        "desired_contour": desired,
        "actual_contour": actual,
        "success_contour": bool(success),
    }
    return entry

def save_accomp_entries_to_file(state, out_dir=EVAL_DIR, prefix="accomp_notes"):
    """
    Write state['note_entries'] list to a JSON file. The session file name is stored in
    state['note_entries_file'] so subsequent notes append to the same file.
    """
    entries = state.get('note_entries', [])
    if not entries:
        return None

    # create session file name once
    if 'note_entries_file' not in state:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.json"
        state['note_entries_file'] = filename

    path = os.path.join(out_dir, state['note_entries_file'])
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(entries, fp, indent=2, ensure_ascii=False)
    except Exception as e:
        # don't crash the audio thread / UI on disk error
        print(f"Error saving accompaniment entries to {path}: {e}")
        return None
    return path