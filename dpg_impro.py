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
from accompaniement import get_pitches_by_chord, chord_loop, make_vlmc_for_chord, transpose_chord_data, play_mp3
from interval import generate_symbols_intervals, build_interval_table, build_interval_dict
from impro_genie import PianoGenieEngine


log = print # type:ignore
# Mapping clavier pour contour mélodique
KEYBOARD_MAPPING = {
    pygame.K_a: 0, pygame.K_z: 1, pygame.K_e: 2, pygame.K_r: 3, 
    pygame.K_t: 4, pygame.K_y: 5, pygame.K_u: 6, pygame.K_i: 7,
}

PROGRESSION = ["A7", "A7", "A7", "A7", "D7", "D7", "A7", "A7", "E7", "D7", "A7", "E7"]
riff = [0, 2, 0, 2, 0, 4, 0, 4]
xml_folder ="/home/sylogue/midi_xml/omnibook_xml"

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
    elif ext == ".mid" or ext == ".midi":
        symbols = MidiSymbolProcessor().process_midi_file(input_path)
        if not symbols:
            raise ValueError(f"No symbols generated for {input_path}")
    elif ext == ".pt":
        pass
    else:
        raise ValueError(f"Extension de corpus non supportée : {ext}")
    return symbols

def load_symbols(input_path: str, mode: str, markov_order: int, similarity_level: int,  xml_folder: str,
        progression: List[str], accomp_info) -> Dict[str, Any]:
    
    if mode == "Autoencoder":
        return {
            'symbols': [],            
            'model_path': input_path, 
            'config_path': "piano_genie/cfg.json"
        }
    
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
        if accomp_info == "normal":
            chord_map = get_pitches_by_chord(xml_folder, progression)
            result['chord_map'] = chord_map
            
            # Build VLMC tables normales (sans intervalles)
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
                #use_intervals=False  # Explicitement False
            )
            
        elif accomp_info == "by chord type":
            chord_map = get_pitches_by_chord(xml_folder, progression, group_by_type=True, max_notes_per_chord=600)
            chord_map = transpose_chord_data(chord_map, PROGRESSION)
            result['chord_map'] = chord_map
            
            # Build VLMC tables normales (sans intervalles)  
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
                #use_intervals=False  # Explicitement False
            )
            
        elif accomp_info == "by interval":
            # 1) récupérer les pitches groupés par type (comme "by chord type")
            chord_map = get_pitches_by_chord(
                xml_folder,
                progression,
                group_by_type=True,
                max_notes_per_chord=600
            )
            chord_map = transpose_chord_data(chord_map, progression)

            # 2) construire le dictionnaire d'intervalles pour chaque accord
            #    (build_interval_dict attend chord_map: chord -> list[pitches])
            interval_dict = build_interval_dict(chord_map)

            # 3) construire la table de transitions d'intervalles (ordre = markov_order)
            #    build_interval_table prend data: dict->sequences et max_order (ordre de backoff)
            interval_table = build_interval_table(interval_dict, max_order=markov_order)

            # 4) sauvegarder dans le résultat pour usage ultérieur (génération)
            result['chord_map'] = chord_map
            result['interval_dict'] = interval_dict
            result['interval_table'] = interval_table

            # 5) construire les VLMC en mode "intervalles"
            #    make_vlmc_for_chord doit accepter use_intervals=True pour construire
            #    ses modèles en travaillant sur les intervalles plutôt que sur les pitches bruts.
            result['vlmcs'] = make_vlmc_for_chord(
                chord_map,
                max_order=markov_order,
                similarity_level=similarity_level,
                use_intervals=True
            )
        
    result['progression'] = progression
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
            contour=True
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        # Markov: contexte variable
        sym, next_prob, top_probs = generate_symbol_vlmc(
        previous_symbols   = state['symbol_history'],
        vlmc_table         = state['vlmc_table'],
        max_order          = config['markov_order'],
        gap                = gap,
        contour            = True,
        similarity_level= config['sim_lvl'],
        n_candidates = config['n_candidat'],
        current_chord=None,  
        use_scale_filter=False
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
        chord_name = PROGRESSION[bar_index % len(PROGRESSION)]
        print(f"[DEBUG]-{chord_name}")
        # grab that chord's VLMC table + all_keys
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        # generate one symbol (note) from *that* table
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            max_order          = config['markov_order'],
            gap                = gap,
            contour            = True,
            similarity_level   = config['sim_lvl'],
            n_candidates = config['n_candidat'],
            current_chord=chord_name,  
            use_scale_filter=True
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

    elif config['mode'] == 'Autoencoder':
        btn_idx = KEYBOARD_MAPPING[event.key]
        # génère une note via le décodeur
        midi_pitch, onset = state["engine"].generate_note_from_button(btn_idx, dur_eff)
        raw_note = [midi_pitch]
        
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
        velocity (int): Vélocité MIDI (0-127)
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
        new_state, raw_note, type_link = generate_note_oracle(
            state['prev_state'],
            state['trans_oracle'],
            state['supply'],
            state['symbols'],
            dur_eff,
            gap,
            p=config['p'],
            contour=True
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        # Markov: contexte variable
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['symbol_history'],
            vlmc_table         = state['vlmc_table'],
            max_order          = config['markov_order'],
            gap                = gap,
            contour            = True,
            similarity_level   = config['sim_lvl'],
            n_candidates       = config['n_candidat'],
            current_chord="C7",  # L'accord actuel de votre progression
            use_scale_filter=True
        )
        # Mise à jour du contexte
        state['symbol_history'].append(sym)
        # Historique pour display
        state['pitch_history'].append(sym['pitch'])
        # Conserver uniquement les derniers N selon markov_order
        if len(state['symbol_history']) > config['markov_order'] + 1:
            state['symbol_history'].pop(0)
            state['pitch_history'].pop(0)
        raw_note = sym

        if log_callback and top_probs:
            chosen = sym['pitch']
            choices = [(s['pitch'], p) for s, p in top_probs]
            log_callback(f"__markov_probs__:{chosen}:{choices}:{next_prob}")

    elif config['mode'] == 'random':
        rnd = random.choice(state['unique_pitches'])
        raw_note = rnd if isinstance(rnd, list) else [rnd]

    elif config["mode"] == "accompagnement":
        # figure out how many bars have elapsed
        elapsed = time() - last_times.get('accomp_start', state['accomp_start'])
        bar_index = int(elapsed / state['bar_dur'])
        chord_name = PROGRESSION[bar_index % len(PROGRESSION)]
        print(f"[DEBUG]-{chord_name}")
        # grab that chord's VLMC table + all_keys
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        # generate one symbol (note) from *that* table
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            max_order          = config['markov_order'],
            gap                = gap,
            contour            = True,
            similarity_level   = config['sim_lvl'],
            n_candidates       = config['n_candidat'],
            current_chord=chord_name, 
            use_scale_filter=True
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

    elif config['mode'] == 'Autoencoder':
        btn_idx = note_index  # Use note_index directly as button index
        # génère une note via le décodeur
        midi_pitch, onset = state["engine"].generate_note_from_button(btn_idx, dur_eff)
        raw_note = [midi_pitch]  # Wrap in list to match expected format

    # Jouer note ou accord
    pitches_to_play, duration, vel = normalize_note(raw_note, dur_eff)
    # Use the MIDI velocity instead of the one from normalize_note
    vel = velocity
    for p in pitches_to_play:
        synth.noteon(0, p, vel)
    
    # Store pitches in note_buffer using note_index as key
    state['note_buffer'][note_index] = pitches_to_play
    
    # Initialize key_start dict if it doesn't exist
    if 'key_start' not in last_times:
        last_times['key_start'] = {}
    last_times['key_start'][note_index] = now
    
    log(f"KD MIDI note {note_index} -> pitch {pitches_to_play}, vel {vel}, dur_eff {dur_eff}, gap {gap}")

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
    """Boucle principale d'improvisation avec gestion d'arrêt améliorée."""
    
    history = []

    global log
    def log(msg):
        if log_callback:
            log_callback(msg)
        history.append(msg)

    try:
        data = load_symbols(
            config['corpus'], config['mode'],
            config.get('markov_order', 1), config.get('sim_lvl', 1), xml_folder, PROGRESSION, config.get("accomp_mod_tag")
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
        random_preset = [0, 11, 12, 16, 18]
        synth_accomp = init_audio(config['sf2_path'], preset=random.choice(random_preset))

        # Attach mode-specific data
        if config['mode'] == 'oracle':
            state['trans_oracle'] = data['trans_oracle']
            state['supply'] = data['supply']
        elif config['mode'] == 'markov':
            state['vlmc_table'] = data['vlmc_table']
            state['notes'] = data['notes']

        elif config['mode'] == 'accompagnement':
            # On récupère directement le chord_map et les VLMCs générés par load_symbols
            state['chord_map'] = data['chord_map']      # { accord: [hauteurs MIDI] }
            state['vlmcs']     = data['vlmcs']          # { accord: (vlmc_table, all_keys) }
            state['progression'] = data['progression']

            # Historiques par accord pour alimenter la génération si besoin
            state['accomp_history'] = {ch: [] for ch in state['chord_map']}
            backtrack_bpm = 90
            beat = 60.0 / backtrack_bpm #config['bpm']
            state['bar_dur']      = 4 * beat
            state['accomp_start'] = time()
            state['accomp_stop'] = threading.Event()
            global _accomp_stop
            _accomp_stop = state['accomp_stop']

            use_backtrack = config['backtrack_mode']
            backtrack_path = '/home/sylogue/stage/metaImpro/corpus/Blues Backing Track in A (90bpm).mp3'
            
            if use_backtrack and backtrack_path and os.path.exists(backtrack_path):
                # Use MP3 backtrack
                if log_callback:
                    log_callback(f"Using MP3 backtrack: {os.path.basename(backtrack_path)} at {backtrack_bpm} BPM")
                
                threading.Thread(
                    target=play_mp3,
                    args=(
                        backtrack_path,
                        state['accomp_stop'],
                        state['progression'],
                        backtrack_bpm
                    ),
                    kwargs={"log_callback": log_callback},
                    daemon=True
                ).start()
            else:
                # Use chord loop instead of backtrack
                if log_callback:
                    log_callback(f"Using chord loop accompaniment")
                
                threading.Thread(
                    target=chord_loop,
                    args=(
                        synth_accomp,
                        state['accomp_stop'],
                        state['progression'],
                    ),
                    kwargs={
                        "bpm": config['bpm'],
                        "velocity": 60,
                        "log_callback": log_callback,
                        #"riff_pattern": riff
                    },
                    daemon=True
                ).start()

        elif config["mode"] == "Autoencoder":
            engine = PianoGenieEngine(
                model_path=data["model_path"],
                config_path=data["config_path"]
            )
            engine.reset_generation()
            state["engine"] = engine

        # Random and Markov need pitches list
        if config['mode'] in ('markov', 'random'):
            state['unique_pitches'] = data['unique_pitches']
        
        
        last_times = {
            'key_start': {}, 'last_note_end': None,
            'last_note_duration': 0.1, 'prev_key_index': None
        }
        
        use_pygame = "Midi Through:Midi Through Port-0 14:0"
        if config['device'] == use_pygame:
            print("Mode clavier Pygame activé (Midi Through détecté)")
            pygame.init()
            pygame.display.set_mode((1, 1))
            
            while not stop_event.is_set():
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or stop_event.is_set():
                        stop_event.set()
                        break
                    elif ev.type == pygame.KEYDOWN and ev.key in KEYBOARD_MAPPING:
                        last_times['key_start'][ev.key] = time()
                        handle_keydown(ev, state, config, synth, history, last_times, log_callback)
                    elif ev.type == pygame.KEYUP and ev.key in last_times['key_start']:
                        handle_keyup(ev, state, synth, history, last_times)
                        
                # Petite pause pour éviter une consommation CPU excessive
                pygame.time.wait(10)
                
            pygame.quit()

        else:
            try:
                midi_port = mido.open_input(config['device']) #type:ignore
                print(f"MIDI mode actif avec le port : {config['device']}")
                
                while not stop_event.is_set():
                    # Utiliser polling avec timeout pour vérifier stop_event
                    if midi_port.poll():
                        msg = midi_port.receive(block=False)
                        if msg.type == 'note_on' and msg.velocity > 0:
                            handle_keydown_midi(msg.note, msg.velocity, state, config, synth, history, last_times, log_callback)
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            handle_keyup_midi(msg.note, state, synth, history, last_times)
                    else:
                        # Petite pause si aucun message MIDI
                        threading.Event().wait(0.01)  # 10ms
                        
                midi_port.close()
                
            except (OSError, IOError) as e:
                if log_callback:
                    log_callback(f"Erreur d'ouverture du port MIDI : {config['device']} - {e}")
                print(f"Erreur d'ouverture du port MIDI : {config['device']} - {e}")

    except Exception as e:
        if log_callback:
            log_callback(f"Erreur dans la boucle d'improvisation: {e}")
        print(f"Erreur dans la boucle d'improvisation: {e}")
    
    finally:
        # Nettoyage final
        try:
            if 'synth' in locals():
                synth.delete()
            if 'synth_accomp' in locals():
                synth_accomp.delete()
            if '_accomp_stop' in globals() and _accomp_stop is not None:
                _accomp_stop.set()
        except Exception as e:
            print(f"Erreur lors du nettoyage: {e}")
        
        if log_callback:
            log_callback("Boucle d'improvisation terminée")

def stop_impro_thread():
    """Arrête le thread d'improvisation en cours."""
    global _impro_thread, _stop_event, _accomp_stop
    
    # Arrêter l'accompagnement si il existe
    if '_accomp_stop' in globals() and _accomp_stop is not None:
        _accomp_stop.set()
    
    # Arrêter le thread principal d'improvisation
    if _stop_event is not None:
        _stop_event.set()
        if _impro_thread is not None and _impro_thread.is_alive():
            _impro_thread.join(timeout=2.0)  # Attendre max 2 secondes

def run_impro(config, log_callback=None):
    """Lance (ou relance) la boucle d'improvisation dans un thread daemon.

    Args:
        config (dict): Configuration d'improvisation.
        log_callback: Fonction de callback pour les logs.
    Returns:
        threading.Thread: Le thread en cours d'exécution (daemon).
    """
    global _impro_thread, _stop_event, _accomp_stop
    
    # Arrêter proprement toute improvisation en cours
    stop_impro_thread()
        
    # Créer un nouvel event et thread
    _stop_event = threading.Event()
    _impro_thread = threading.Thread(target=improvisation_loop,
                                     args=(config, _stop_event, log_callback),
                                     daemon=True)
    _impro_thread.start()
    return _impro_thread
