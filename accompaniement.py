from music21 import * # type:ignore
from typing import List
from time import sleep
from markov import build_vlmc_table, generate_symbol_vlmc, symbol_to_key, truncate_key


def get_pitches_by_chord():
    # 1) Load the Omnibook XML and extract pitch lists
    xml_file = "/home/sylogue/midi_xml/omnibook_xml/Omnibook xml/KC_Blues.xml"
    s = converter.parse(xml_file)

    # all notes whose *harmony* context is exactly C7
    c7_notes = [
        n for n in s.recurse().notes
        if n.getContextByClass(harmony.ChordSymbol)
        and n.getContextByClass(harmony.ChordSymbol).figure == 'C7' #type:ignore
    ]

    # extract raw MIDI pitches
    c7_pitches = [n.pitch.midi for n in c7_notes if isinstance(n, note.Note)]
    # for C#7, just transpose each by +1 semitone
    c7_sharp_pitches = [p + 1 for p in c7_pitches]
    return c7_pitches, c7_sharp_pitches

def chord_loop(synth,
               stop_event,
               bpm: int = 120,
               velocity: int = 80,
               lower_octave: int = 12,
               log_callback=None):
    """
    Play an endless piano-stride 4/4 loop over C7 and C#7.

    On each bar:
      - Beats 1 & 3: left-hand bass (root one octave down)
      - Beats 2 & 4: left-hand chord (root position)

    Args:
      synth:      fluidsynth.Synth already initialized.
      stop_event: threading.Event to break the loop.
      bpm:        beats per minute.
      velocity:   MIDI velocity for all notes.
      lower_octave: number of semitones to transpose left-hand bass down.
      log_callback: optional fn(str) to receive log messages.
    """
    roots = [60, 61]  # C and C#
    fifths = [67, 68]  # G and G# (a perfect fifth above the root)
    chords = [
        [64, 67, 70],   # C7 (E, G, Bb)
        [65, 68, 71]    # C#7 (F, G#, B)
    ]
    chord_names = ["C7", "C#7"]
    beat_dur = 120.0 / bpm
    # stride pattern: bass, chord, fifth, chord
    pattern = ["bass", "chord", "fifth", "chord"]
    idx = 0

    while not stop_event.is_set():
        chord_idx = idx % 2
        root = roots[chord_idx]
        fifth = fifths[chord_idx]
        chord = chords[chord_idx]

        for step, part in enumerate(pattern, start=1):
            if stop_event.is_set():
                break
            notes_to_play = []
            if part == "bass":
                # play root one octave down
                notes_to_play = [root - lower_octave]
            elif part == "fifth":
                # play fifth one octave down
                notes_to_play = [fifth - lower_octave]
            else:
                # play full chord in LH
                notes_to_play = chord
                print(f"[DEBUG] {chord_names[chord_idx]}")
            for p in notes_to_play:
                synth.noteon(0, p, velocity)
            if log_callback:
                log_callback(f"Stride: Bar {idx+1}, beat {step}: {part} → {notes_to_play}")

            # hold for one beat
            sleep(beat_dur)

            # note off for this part
            for p in notes_to_play:
                synth.noteoff(0, p)

        idx += 1

def make_vlmc_for_chord(symbol_sequences, max_order=3, similarity_level=1):
    """
    Given a dict mapping chord name → list of symbols (notes/chords) from your corpus,
    build a VLMC table and collect all possible keys.
    Returns a dict: chord_name → (vlmc_table, all_keys).
    """
    vlmcs = {}
    
    for chord_name, seq in symbol_sequences.items():
        # 1) build the VLMC table over the raw sequence
        table = build_vlmc_table(seq,
                                 max_order=max_order,
                                 similarity_level=similarity_level)

        # 2) build your fallback key‑list from the raw symbols
        #    (so even symbols that never occur as "successors" are included)
        keyset = {
            truncate_key(symbol_to_key(sym), similarity_level)
            for sym in seq
        }
        all_keys = list(keyset)

        vlmcs[chord_name] = (table, all_keys)

    return vlmcs