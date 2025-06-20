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
               log_callback=None):
    """
    Play an endless 4/4 loop alternating C7 and C#7.

    Args:
      synth:    fluidsynth.Synth already initialized.
      stop_event: threading.Event to break the loop.
      bpm:      beats per minute (default 120).
      velocity: MIDI velocity for all chord notes.
      log_callback: optional fn(str) to receive log messages.
    """

    c7_chord, c7_sharp_chord = [60,64,67,70], [61,65,68,71]

    beat_dur = 60.0 / bpm
    bar_dur  = beat_dur * 4
    chords   = [c7_chord, c7_sharp_chord]
    idx = 0

    while not stop_event.is_set():
        chord = chords[idx % 2]
        # Note‐on all voices
        for p in chord:
            synth.noteon(0, p, velocity)
        if log_callback:
            log_callback(f"Accompaniment: playing {'C7' if idx%2==0 else 'C#7'} → {chord}")
        # hold for one bar
        sleep(bar_dur)
        # Note‐off
        for p in chord:
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