from time import time
import mido
import fluidsynth
import random
import numpy as np
from typing import List, Any, Dict
import pygame
import threading
import os
import json
from factor_oracle import OracleBuilder, generate_note_oracle
from midi_processor import MidiSymbolProcessor
from markov import build_vlmc_table, generate_symbol_vlmc, symbol_to_key
from accompaniment import chord_loop, make_vlmc_for_chord, get_pitches_by_chord


log = print # type:ignore
# Mapping clavier pour contour mélodique
KEYBOARD_MAPPING = {
    pygame.K_a: 0, pygame.K_z: 1, pygame.K_e: 2, pygame.K_r: 3, 
    pygame.K_t: 4, pygame.K_y: 5, pygame.K_u: 6, pygame.K_i: 7,
}

# Variables globales pour gérer le thread d'impro
_impro_thread = None
_stop_event = None

def init_audio(sf2_path: str, driver: str = "pulseaudio", preset: int = 1):
    """
    Initialise FluidSynth avec la SoundFont spécifiée.

    Args:
        sf2_path (str): Chemin vers le fichier SoundFont (.sf2).
        driver (str): Nom du pilote audio (ex: "pulseaudio").
        preset (int): Numéro du preset à sélectionner dans la banque.

    Returns:
        fluidsynth.Synth: L'objet Synth initialisé et prêt à jouer.
    """
    fs = fluidsynth.Synth()
    fs.start(driver=driver)
    sfid = fs.sfload(sf2_path)
    fs.program_select(0, sfid, 0, preset)
    return fs

def load_corpus(input_path: str) -> List[dict]:
    """
    Charge et renvoie la liste des symboles Midi traités.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.json':
        with open(input_path, 'r') as f:
            symbols = json.load(f)
        if not isinstance(symbols, list):
            raise ValueError("JSON must contain a list of symbols")
    else:
        symbols = MidiSymbolProcessor().process_midi_file(input_path)
        if not symbols:
            raise ValueError(f"No symbols generated for {input_path}")
    return symbols

def load_symbols(input_path: str, mode: str, markov_order: int, similarity_level: int) -> Dict[str, Any]:
    symbols = load_corpus(input_path)
    result: Dict[str, Any] = {'symbols': symbols}

    if mode == 'oracle':
        trans, supp = OracleBuilder.build_oracle(symbols)[::2], OracleBuilder.build_oracle(symbols)[1::2]
        # Unpack correctly based on similarity level
        t3, s3, t2, s2, t1, s1 = trans[0], supp[0], trans[1], supp[1], trans[2], supp[2]
        result['trans_oracle'] = {3: t3, 2: t2, 1: t1}[similarity_level]
        result['supply'] = {3: s3, 2: s2, 1: s1}[similarity_level]

    if mode == 'markov':
        vlmc_table = build_vlmc_table(symbols, max_order=markov_order, similarity_level=similarity_level)
        all_keys = list({symbol_to_key(s) for s in symbols})
        result['vlmc_table'] = vlmc_table
        result['notes'] = all_keys

    if mode in ('markov', 'random'):
        # On recalcule la liste des hauteurs disponibles
        result['unique_pitches'] = []
        seen = set()
        for s in symbols:
            if s['type'] == 'note':
                p = s['pitch']
                if p not in seen:
                    result['unique_pitches'].append(p)
                    seen.add(p)
            elif s['type'] == 'chord':
                result['unique_pitches'].append(tuple(s['pitch']))
    if mode == "accompagnement":
        c7_corpus, csharp7_corpus = get_pitches_by_chord()
        result['c7_corpus']     = c7_corpus
        result['csharp7_corpus'] = csharp7_corpus
        
        # build separate VLMC tables per chord
        vlmcs = make_vlmc_for_chord(
            {'C7': c7_corpus, 'C#7': csharp7_corpus},
            max_order=markov_order,
            similarity_level=similarity_level
        )
        result['vlmcs'] = vlmcs
    return result

def normalize_note(note, dur_eff=None, default_velocity=120):
    """
    Turn any of:
      - an int (oracle single note)
      - a 2-tuple (pitch, velocity) or (pitch, duration)
      - a tuple of pitches (a chord)
      - a dict from VLMC: {'type':'note'|'chord', ...}
    into a unified (pitches_list, duration, velocity) triple.
    """
    
    # note = list (oracle generation/random)
    if isinstance(note, list) and all(isinstance(p, int) for p in note):

        return note, dur_eff, default_velocity
    
    # note dict :{'type': 'note', 'pitch': int, 'duration': int, 'velocity': int} (Markov generation)
    if isinstance(note, dict):
        typ = note.get('type')
        if typ == 'note':
            # single pitch wrapped in a list
            return [note['pitch']], \
                    note.get('duration', dur_eff), \
                    note.get('velocity', default_velocity)
        elif typ == 'chord':
            # multiple pitches
            pitches = note['pitch']
            return list(pitches), \
                    note.get('duration', dur_eff), \
                    note.get('velocity', default_velocity)

    raise ValueError(f"Cannot normalize note: {note!r}")

def handle_keydown(event, state, config, synth, history, last_times, log_callback=None):
    """
    Gère un événement KEYDOWN pour générer et jouer une note d'improvisation.

    Args:
        event: pygame KEYDOWN event
        state: dict, contient prev_state, context, note_buffer, etc.
        config: dict, configuration d'impro (mode, p, default_velocity)
        synth: fluidsynth Synth
        history: list, accumulate logs
        last_times: dict, gère key_start, last_note_end, last_note_duration, prev_key_index

    Returns:
        None
    """
    # Calcul du gap
    idx = KEYBOARD_MAPPING[event.key]
    prev_idx = last_times['prev_key_index']
    gap = 0 if prev_idx is None else idx - prev_idx
    last_times['prev_key_index'] = idx

    # Durée effective
    now = time()
    if last_times['last_note_end'] is not None and now >= last_times['last_note_end']:
        silence = now - last_times['last_note_end']
        dur_eff = last_times['last_note_duration'] + silence
    else:
        dur_eff = last_times['last_note_duration']

    # Génération de la note selon mode
    if config['mode'] == 'oracle':
        new_state, raw_note, type_link = generate_note_oracle(
            state['prev_state'],
            state['trans_oracle'],
            state['supply'],
            state['symbols'],
            dur_eff,
            gap,
            p=config['p'],
            contour=config['contour']
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        # Markov: contexte variable
        sym, next_prob, top_probs = generate_symbol_vlmc(
        previous_symbols   = state['symbol_history'],
        vlmc_table         = state['vlmc_table'],
        all_keys           = state['notes'],
        max_order          = config['markov_order'],
        gap                = gap,
        contour            = config['contour'],
        similarity_level= config['sim_lvl']
        )
        # Mise à jour du contexte
        state['symbol_history'].append(sym)
        #historique pour display
        state['pitch_history'].append(sym['pitch'])
        # Conserver uniquement les derniers N selon markov_order
        if len(state['symbol_history']) > config['markov_order'] + 1:
            state['symbol_history'].pop(0)
            state['pitch_history'].pop(0)
        raw_note = sym

        if log_callback and top_probs:
            chosen = sym['pitch']
            choices = [(s['pitch'], p) for s,p in top_probs]
            log_callback(f"__markov_probs__:{chosen}:{choices}:{next_prob}")
            
    elif config['mode'] == 'random':
        rnd = random.choice(state['unique_pitches'])
        raw_note = rnd if isinstance(rnd, list) else [rnd]
    
    elif config["mode"] == "accompagnement":
        # figure out how many bars have elapsed
        elapsed = time() - last_times.get('accomp_start', state['accomp_start'])
        bar_index = int(elapsed / state['bar_dur'])
        chord_name = "C7" if (bar_index % 2) == 0 else "C#7"

        # grab that chord's VLMC table + all_keys
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        # generate one symbol (note) from *that* table
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            all_keys           = all_keys,
            max_order          = config['markov_order'],
            gap                = gap,
            contour            = config['contour'],
            similarity_level   = config['sim_lvl']
        )

        # update that chord's own history
        state['accomp_history'][chord_name].append(sym)
        # (optional: keep it bounded by markov_order)
        if len(state['accomp_history'][chord_name]) > config['markov_order'] + 1:
            state['accomp_history'][chord_name].pop(0)

        raw_note = sym

        if log_callback:
            choices = [(s['pitch'], p) for s, p in top_probs]
            log_callback(f"__markov_probs__:{sym['pitch']}:{choices}:{next_prob}")

    # Jouer note ou accord
    pitches_to_play, duration, vel = normalize_note(raw_note, dur_eff)
    for p in pitches_to_play:
        synth.noteon(0, p, vel)
    state['note_buffer'][event.key] = pitches_to_play
    log(f"KD {pygame.key.name(event.key)} -> pitch {pitches_to_play}, vel {vel}, dur_eff {dur_eff}, gap {gap}")
    



def handle_keyup(event, state, synth, history, last_times):
    """
    Gère un événement KEYUP pour arrêter la note et enregistrer sa durée.

    Args:
        event: pygame KEYUP event
        state: dict, contient note_buffer
        synth: fluidsynth Synth
        history: list, accumulate logs
        last_times: dict, gère key_start, last_note_end, last_note_duration

    Returns:
        None
    """
    # Durée réelle
    dur = time() - last_times['key_start'][event.key]
    # Récupérer pitch et arrêter le son
    pitches = state['note_buffer'].pop(event.key, None)
    if pitches is not None:
        for p in pitches:
            synth.noteoff(0, p)
    info = f"KU {pygame.key.name(event.key)} -> pitch {pitches}, dur {dur:.2f}"
    log(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][event.key]

def handle_keydown_midi(note_index, velocity, state, config, synth, history, last_times, log_callback=None):
    """
    Gère un événement note_on MIDI pour générer et jouer une note d'improvisation.

    Args:
        note_index (int): Index de la note MIDI (0-127)
        state (dict): Contient prev_state, context, note_buffer, etc.
        config (dict): Configuration d'impro (mode, p, default_velocity)
        synth (fluidsynth.Synth): Synthétiseur FluidSynth
        history (list): Accumule les logs
        last_times (dict): Gère key_start, last_note_end, last_note_duration, prev_key_index

    Returns:
        None
    """
    # Calcul du gap
    prev_idx = last_times.get('prev_key_index')
    gap = 0 if prev_idx is None else note_index - prev_idx
    last_times['prev_key_index'] = note_index

    # Durée effective
    now = time()
    if last_times['last_note_end'] is not None and now >= last_times['last_note_end']:
        silence = now - last_times['last_note_end']
        dur_eff = last_times['last_note_duration'] + silence
    else:
        dur_eff = last_times['last_note_duration']

    # Génération de la note selon le mode
    if config['mode'] == 'oracle':
        new_state, raw_note, _ = generate_note_oracle(
            state['prev_state'],
            state['trans_oracle'],
            state['supply'],
            state['symbols'],
            dur_eff,
            gap,
            p=config['p'],
            contour=config['contour']
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        # Markov: contexte variable
        sym, next_prob, top_probs = generate_symbol_vlmc(
        previous_symbols   = state['symbol_history'],
        vlmc_table         = state['vlmc_table'],
        all_keys           = state['notes'],
        max_order          = config['markov_order'],
        gap                = gap,
        contour            = config['contour'],
        similarity_level= config['sim_lvl']
        )
        # Mise à jour du contexte
        state['symbol_history'].append(sym)
        # Conserver uniquement les derniers N selon markov_order
        max_ord = config['markov_order']
        if max_ord > 0 and len(state['symbol_history']) > max_ord:
            state['symbol_history'] = state['symbol_history'][-max_ord:]
        raw_note = sym

        if log_callback and top_probs:
            p = sym['pitch']
            choices = [(s['pitch'], prob) for s, prob in top_probs]
            log_callback(f"__markov_probs__:{p}:{choices}:{next_prob}")

    elif config['mode'] == 'random':
        rnd = random.choice(state['unique_pitches'])
        raw_note = rnd if isinstance(rnd, list) else [rnd]

    elif config["mode"] == "accompagnement":
        # how many bars have passed since we started
        elapsed  = time() - state['accomp_start']
        bar_index = int(elapsed / state['bar_dur'])
        chord_name = "C7" if (bar_index % 2) == 0 else "C#7"

        # lookup that chord’s VLMC table + keys
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        # generate a symbol from *that* model
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            all_keys           = all_keys,
            max_order          = config['markov_order'],
            gap                = gap,
            contour            = config['contour'],
            similarity_level   = config['sim_lvl']
        )

        # update only this chord’s history
        h = state['accomp_history'][chord_name]
        h.append(sym)
        if len(h) > config['markov_order'] + 1:
            h.pop(0)

        raw_note = sym

        if log_callback:
            choices = [(s['pitch'], prob) for s, prob in top_probs]
            log_callback(f"__markov_probs__:{sym['pitch']}:{choices}:{next_prob}")

    # Jouer la note ou accord
    pitches_to_play, duration, _ = normalize_note(raw_note, dur_eff, default_velocity=velocity)
    vel = velocity
    for p in pitches_to_play:
        synth.noteon(0, p, vel)            
        
    state['note_buffer'][note_index] = pitches_to_play
    last_times['key_start'][note_index] = now
    log(
        f"KD MIDI note {note_index} -> pitch {pitches_to_play}, vel {vel}, dur_eff {dur_eff}, gap {gap}"
    )

def handle_keyup_midi(note_index, state, synth, history, last_times):
    """
    Gère un événement note_off MIDI pour arrêter la note et enregistrer sa durée.

    Args:
        note_index (int): Index de la note MIDI (0-127)
        state (dict): Contient note_buffer
        synth (fluidsynth.Synth): Synthétiseur FluidSynth
        history (list): Accumule les logs
        last_times (dict): Gère key_start, last_note_end, last_note_duration

    Returns:
        None
    """
    # Vérifier si la note était en cours
    start_time = last_times['key_start'].get(note_index)
    if start_time is None:
        return  # Ignorer si aucune note_on correspondante

    # Durée réelle
    dur = time() - start_time
    # Récupérer le pitch et arrêter le son
    pitches = state['note_buffer'].pop(note_index, None)
    if pitches is not None:
        for p in pitches:
            synth.noteoff(0, p)
    info = f"KU MIDI note {note_index} -> pitch {pitches}, dur {dur:.2f}"
    log(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][note_index]

    
def improvisation_loop(config, stop_event, log_callback=None):

    history = []

    global log
    def log(msg):
        if log_callback:
            log_callback(msg)
        history.append(msg)

    data = load_symbols(
        config['corpus'], config['mode'],
        config.get('markov_order', 1), config.get('sim_lvl', 1)
    )

    symbols = data['symbols']
    initial = symbols[0] if symbols else {'type': 'note', 'pitch': 60, 'duration': 0, 'velocity': 110}
    
    state: Dict[str, Any] = {
        'prev_state':    0,
        'symbol_history':[initial],
        'pitch_history': [initial['pitch'] if initial['type']=='note' else initial['pitch'][0]],
        'symbols':       symbols,
        'note_buffer':   {}
    }

    synth = init_audio(config['sf2_path'])


    # Attach mode-specific data
    if config['mode'] == 'oracle':
        state['trans_oracle'] = data['trans_oracle']
        state['supply'] = data['supply']
    elif config['mode'] == 'markov':
        state['vlmc_table'] = data['vlmc_table']
        state['notes'] = data['notes']

    elif config['mode'] == 'accompagnement':
        # instead of blocking chord_loop, prepare VLMC tables once:
        state['vlmcs'] = make_vlmc_for_chord({
            "C7":  data['c7_corpus'],
            "C#7": data['csharp7_corpus'],}, 
            max_order=config['markov_order'],
        similarity_level=config['sim_lvl'])

        # per‑chord histories for context
        state['accomp_history'] = {"C7": [], "C#7": []}

        # timing to figure out which bar we’re in
        state['accomp_start'] = time()
        beat = 60.0 / 120
        state['bar_dur'] = 4 * beat
        state['accomp_stop'] = threading.Event()
        global _accomp_stop
        _accomp_stop = state['accomp_stop']

        threading.Thread(
            target=chord_loop,
            args=(synth, state['accomp_stop']),
            kwargs={
                "bpm": 120,
                "velocity": 50,
                "log_callback": log_callback
            },
            daemon=True
        ).start()
    # Random and Markov need pitches list
    if config['mode'] in ('markov', 'random'):
        state['unique_pitches'] = data['unique_pitches']
    
    
    last_times = {
        'key_start': {}, 'last_note_end': None,
        'last_note_duration': 0.1, 'prev_key_index': None
    }
    use_pygame = "Midi Through:Midi Through Port-0 14:0"
    if config['device'] == use_pygame:
        print("🎹 Mode clavier Pygame activé (Midi Through détecté)")
        pygame.init()
        pygame.display.set_mode((1, 1))
        while not stop_event.is_set():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stop_event.set()
                elif ev.type == pygame.KEYDOWN and ev.key in KEYBOARD_MAPPING:
                    last_times['key_start'][ev.key] = time()
                    handle_keydown(ev, state, config, synth, history, last_times, log_callback)
                elif ev.type == pygame.KEYUP and ev.key in last_times['key_start']:
                    handle_keyup(ev, state, synth, history, last_times)
        pygame.quit()

    else:
        try:
            midi_port = mido.open_input(config['device']) #type:ignore
            print(f"✅ MIDI mode actif avec le port : {config['device']}")
            for msg in midi_port:
                if stop_event.is_set():
                    break
                if msg.type == 'note_on' and msg.velocity > 0:
                    handle_keydown_midi(msg.note, msg.velocity, state, config, synth, history, last_times, log_callback)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    handle_keyup_midi(msg.note, state, synth, history, last_times)
            midi_port.close()
        except (OSError, IOError):
            print(f"❗ Erreur d'ouverture du port MIDI : {config['device']}")

        pygame.quit()

    if config['mode'] == 'accompagnement':
        state['accomp_stop'].set()

    synth.delete()


def run_impro(config, log_callback=None):
    """Lance (ou relance) la boucle d'improvisation dans un thread daemon.

    Args:
        config (dict): Configuration d'improvisation.
    Returns:
        threading.Thread: Le thread en cours d'exécution (daemon).
    """
    global _impro_thread, _stop_event, _accomp_stop
    # On arrête la chord loop
    if '_accomp_stop' in globals() and _accomp_stop is not None:
        _accomp_stop.set()

    # Arrêter le thread existant si nécessaire
    if _stop_event is not None:
        _stop_event.set()
        if _impro_thread is not None:
            _impro_thread.join(timeout=1)

    # Créer un nouvel event et thread
    _stop_event = threading.Event()
    _impro_thread = threading.Thread(target=improvisation_loop,
                                     args=(config, _stop_event, log_callback),
                                     daemon=True)
    _impro_thread.start()
    return _impro_thread
