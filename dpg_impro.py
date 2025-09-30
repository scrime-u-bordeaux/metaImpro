from time import time, sleep
import mido
import fluidsynth
import random
import numpy as np
from typing import List, Any, Dict
import pygame
import threading
import os
import json
import pretrained as pt
from factor_oracle import generate_note_oracle
from markov import generate_symbol_vlmc, is_white_note
from accompaniement import chord_loop, play_mp3
from record_impro import serialize_info, save_accomp_entries_to_file, _sign_to_str
#from impro_genie import PianoGenieEngine


log = print # type:ignore
# Mapping clavier pour contour mélodique
KEYBOARD_MAPPING = {
    pygame.K_q: 0, pygame.K_s: 2, pygame.K_d: 4, pygame.K_f: 6, 
    pygame.K_g: 8, pygame.K_h: 10, pygame.K_j: 12, pygame.K_k: 14,
}

BLACK_KEY_MAPPING = {
    pygame.K_a: 1, pygame.K_z: 3, pygame.K_e: 5, pygame.K_r: 7,
    pygame.K_t: 9, pygame.K_y: 11, pygame.K_u: 13, pygame.K_i: 15,
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
    Updated: uses sym['played_pitch'] for playback (if present) while keeping sym
    (logical pitch) in the history (Option A).
    """
    # Calcul du gap & mapping
    if event.key in KEYBOARD_MAPPING:
        idx = KEYBOARD_MAPPING[event.key]
        is_black = False
    elif event.key in BLACK_KEY_MAPPING:
        idx = BLACK_KEY_MAPPING[event.key]
        is_black = True
    else:
        # key not bound -> ignore
        return

    # scale filter
    use_scale_filter = not is_black

    prev_idx = last_times.get('prev_key_index')
    gap = 0 if prev_idx is None else idx - prev_idx
    last_times['prev_key_index'] = idx

    # Durée effective
    now = time()
    if last_times.get('last_note_end') is not None and now >= last_times.get('last_note_end'):
        silence = now - last_times.get('last_note_end')
        dur_eff = last_times.get('last_note_duration', 0.0) + silence
    else:
        dur_eff = last_times.get('last_note_duration', 0.0)

    # Génération de la note selon mode
    raw_note = None

    if config['mode'] == 'oracle':
        new_state, raw_note, type_link = generate_note_oracle(
            state['prev_state'],
            state['trans_oracle'],
            state['supply'],
            state['symbols'],
            dur_eff,
            gap,
            p=config.get('p', 0.5),
            contour=True
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['symbol_history'],
            vlmc_table         = state['vlmc_table'],
            max_order          = config.get('markov_order', 3),
            gap                = gap,
            contour            = True,
            similarity_level   = config.get('sim_lvl', 3),
            n_candidates       = config.get('n_candidat', 1),
            current_chord      = None,
            use_scale_filter   = False,
        )
        # Keep logical symbol in history (Option A)
        state['symbol_history'].append(sym)
        state['pitch_history'].append(sym['pitch'])
        if len(state['symbol_history']) > config.get('markov_order', 3) + 1:
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
        elapsed = time() - last_times.get('accomp_start', state.get('accomp_start', 0.0))
        bar_index = int(elapsed / state.get('bar_dur', 1.0))
        chord_name = PROGRESSION[bar_index % len(PROGRESSION)]
        print(f"[DEBUG]-{chord_name}")
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            max_order          = config.get('markov_order', 3),
            gap                = gap,
            contour            = True,
            similarity_level   = config.get('sim_lvl', 3),
            n_candidates       = config.get('n_candidat', ),
            current_chord      = chord_name,
            use_scale_filter   = use_scale_filter,
            force_out_of_scale = not use_scale_filter,
        )

        state['accomp_history'][chord_name].append(sym)
        if len(state['accomp_history'][chord_name]) > config.get('markov_order', 3) + 1:
            state['accomp_history'][chord_name].pop(0)

        raw_note = sym

        if log_callback:
            choices = [(s['pitch'], p) for s, p in top_probs]
            log_callback(f"__markov_probs__:{sym['pitch']}:{choices}:{next_prob}")

    elif config['mode'] == 'Autoencoder':
        btn_idx = KEYBOARD_MAPPING[event.key]
        midi_pitch, onset = state["engine"].generate_note_from_button(btn_idx, dur_eff)
        raw_note = [midi_pitch]

    # Jouer note ou accord
    # If raw_note is the symbol dict produced by generate_symbol_vlmc, prefer its 'played_pitch'.
    if isinstance(raw_note, dict) and 'played_pitch' in raw_note:
        played = raw_note['played_pitch']
        if not isinstance(played, (list, tuple)):
            played = [played]
        pitches_to_play = [int(max(0, min(127, p))) for p in played]
        duration = raw_note.get('duration', dur_eff)
        vel = raw_note.get('velocity', config.get('default_velocity', 100))
    else:
        # fallback to existing normalize_note to handle ints/lists/dicts without played_pitch
        pitches_to_play, duration, vel = normalize_note(raw_note, dur_eff)

    for p in pitches_to_play:
        p_clamped = int(max(0, min(127, p)))
        synth.noteon(0, p_clamped, vel)

    # Store in buffer keyed by the pygame event.key
    state['note_buffer'][event.key] = pitches_to_play

    # mark key start time
    last_times.setdefault('key_start', {})[event.key] = now

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
    Updated to use sym['played_pitch'] for playback while keeping sym in history (Option A).
    """
    # --- initialisations sûres pour variables optionnelles ---
    chord_name = None
    is_black = None
    duration = None
    elapsed = None       
    next_prob = None

    # Calcul du gap
    prev_idx = last_times.get('prev_key_index')
    gap = 0 if prev_idx is None else note_index - prev_idx
    last_times['prev_key_index'] = note_index

    # Durée effective
    now = time()
    if last_times.get('last_note_end') is not None and now >= last_times.get('last_note_end'):
        silence = now - last_times.get('last_note_end')
        dur_eff = last_times.get('last_note_duration', 0.0) + silence
    else:
        dur_eff = last_times.get('last_note_duration', 0.0)

    raw_note = None


    if config['mode'] == 'oracle':
        new_state, raw_note, type_link = generate_note_oracle(
            state['prev_state'],
            state['trans_oracle'],
            state['supply'],
            state['symbols'],
            dur_eff,
            gap,
            p=config.get('p', 0.5),
            contour=True
        )
        state['prev_state'] = new_state
        if log_callback:
            log_callback(f"__progress__:{new_state}:{len(state['symbols'])}")

    elif config['mode'] == 'markov':
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['symbol_history'],
            vlmc_table         = state['vlmc_table'],
            max_order          = config.get('markov_order', 3),
            gap                = gap,
            contour            = True,
            similarity_level   = config.get('sim_lvl', 3),
            n_candidates       = config.get('n_candidat', 1),
            current_chord      = "C7",
            use_scale_filter   = True
        )
        # Keep logical symbol in history (Option A)
        state['symbol_history'].append(sym)
        state['pitch_history'].append(sym['pitch'])
        if len(state['symbol_history']) > config.get('markov_order', 3) + 1:
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
        elapsed = time() - last_times.get('accomp_start', state.get('accomp_start', 0.0))
        bar_index = int(elapsed / state.get('bar_dur', 1.0))
        chord_name = PROGRESSION[bar_index % len(PROGRESSION)]
        print(f"[DEBUG]-{chord_name}")
        vlmc_table, all_keys = state['vlmcs'][chord_name]

        is_white_key = is_white_note(note_index)
        is_black = not is_white_key
        sym, next_prob, top_probs = generate_symbol_vlmc(
            previous_symbols   = state['accomp_history'][chord_name],
            vlmc_table         = vlmc_table,
            max_order          = config.get('markov_order', 2),
            gap                = gap,
            contour            = True,
            similarity_level   = config.get('sim_lvl', 2),
            n_candidates       = config.get('n_candidat', 5),
            current_chord      = chord_name,
            use_scale_filter   = is_white_key,
            force_out_of_scale = is_black,
        )

        state['accomp_history'][chord_name].append(sym)
        if len(state['accomp_history'][chord_name]) > config.get('markov_order', 3) + 1:
            state['accomp_history'][chord_name].pop(0)

        raw_note = sym

        if log_callback:
            choices = [(s['pitch'], p) for s, p in top_probs]
            log_callback(f"__markov_probs__:{sym['pitch']}:{choices}:{next_prob}")

    elif config['mode'] == 'Autoencoder':
        btn_idx = note_index
        midi_pitch, onset = state["engine"].generate_note_from_button(btn_idx, dur_eff)
        raw_note = [midi_pitch]

    # Jouer note ou accord
    # Prefer sym['played_pitch'] if available; otherwise fall back to normalize_note
    if isinstance(raw_note, dict) and 'played_pitch' in raw_note:
        played = raw_note['played_pitch']
        if not isinstance(played, (list, tuple)):
            played = [played]
        pitches_to_play = [int(max(0, min(127, p))) for p in played]
        duration = raw_note.get('duration', dur_eff)
        vel = raw_note.get('velocity', config.get('default_velocity', 100))
        # But per MIDI handler requirements, use the incoming MIDI velocity
        vel = velocity
    else:
        pitches_to_play, duration, vel = normalize_note(raw_note, dur_eff)
        # normalize_note may renvoyer None pour duration; on le laisse tel quel
        vel = velocity  # override with actual MIDI velocity

    # Défensive: s'assurer que pitches_to_play est une liste non vide
    if pitches_to_play is None:
        pitches_to_play = []
    else:
        # clamp again par sécurité
        pitches_to_play = [int(max(0, min(127, p))) for p in pitches_to_play]

    for p in pitches_to_play:
        p_clamped = int(max(0, min(127, p)))
        synth.noteon(0, p_clamped, vel)

    # Store pitches in note_buffer using note_index as key
    state.setdefault('note_buffer', {})[note_index] = pitches_to_play

    # Initialize key_start dict if it doesn't exist
    last_times.setdefault('key_start', {})[note_index] = now

    log(f"KD MIDI note {note_index} -> pitch {pitches_to_play}, vel {vel}, dur_eff {dur_eff}, gap {gap}")

    if prev_idx is None:
        desired_str = None
    else:
        g = int(np.sign(gap)) if gap is not None else 0
        desired_str = _sign_to_str(int(g))
    # Construire l'entrée en protégeant les variables optionnelles
    entry = {
        'chord': chord_name,
        'pitch': pitches_to_play,
        'onset': round(elapsed, 2) if elapsed is not None else None,
        'duration': round(duration, 2) if duration is not None else None,   # durée théorique si fournie
        'velocity': vel,
        'effective_duration': round(dur_eff, 2),
        'note_index': note_index,
        'is_black': is_black,
        'desired': desired_str,
        'actual': None,
        'success': None,
        'note_prob': next_prob
    }

    # Calculer 'actual' de façon sûre (s'il y a un historique de pitchs et au moins une pitch jouée)
    if pitches_to_play and state.get('pitch_history'):
        try:
            last_pitch = state['pitch_history'][-1]
            entry['actual'] = int(np.sign(pitches_to_play[0] - last_pitch))
        except Exception:
            entry['actual'] = None
    else:
        entry['actual'] = None

    # Calculer 'success' de façon sûre :
    # - si pas de prev_idx (pas de note précédente) -> None ou True selon préférence ; ici on met None pour indiquer indéterminé
    # - si pas d'historique de pitchs -> None
    # - sinon comparer le signe
    if prev_idx is None or not state.get('pitch_history'):
        entry['success'] = None
    else:
        if entry['actual'] is None:
            entry['success'] = None
        else:
            desired_sign = int(np.sign(gap))
            entry['success'] = (desired_sign == entry['actual'])

    state.setdefault('note_entries', []).append(entry)


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
        data = pt.load_symbols_cached(
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
            'note_buffer':   {},
            'note_entries':  []
        }

        synth = init_audio(config['sf2_path'], config['audio_driver'])
        random_preset = [0, 11, 12, 16, 18]
        synth_accomp = init_audio(config['sf2_path'], config['audio_driver'],preset=random.choice(random_preset))

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
            """engine = PianoGenieEngine(
                model_path=data["model_path"],
                config_path=data["config_path"]
            )
            engine.reset_generation()
            state["engine"] = engine"""

        # Random and Markov need pitches list
        if config['mode'] in ('markov', 'random'):
            state['unique_pitches'] = data['unique_pitches']
        
        
        last_times = {
            'key_start': {}, 'last_note_end': None,
            'last_note_duration': 0.1, 'prev_key_index': None
        }
        
        use_pygame = "Midi Through:Midi Through Port-0 14:0"
        if config['device_in'] == use_pygame:
            print("Mode clavier Pygame activé (Midi Through détecté)")
            pygame.init()
            pygame.display.set_mode((1, 1))
            
            while not stop_event.is_set():
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or stop_event.is_set():
                        stop_event.set()
                        break
                    elif ev.type == pygame.KEYDOWN and (ev.key in KEYBOARD_MAPPING or ev.key in BLACK_KEY_MAPPING):
                        last_times['key_start'][ev.key] = time()
                        handle_keydown(ev, state, config, synth, history, last_times, log_callback)
                    elif ev.type == pygame.KEYUP and ev.key in last_times['key_start']:
                        handle_keyup(ev, state, synth, history, last_times)
                        
                # Petite pause pour éviter une consommation CPU excessive
                pygame.time.wait(10)
                
            pygame.quit()

        else:
            try:
                default_gesture = {
                    'note_indices': [
                        60, 61, 61, 58, 57, 58, 60, 65, 68, 69, 70, 71, 72, 71, 69
                    ],
                    'velocities': [
                        80, 95, 85, 100, 90, 95, 110, 100, 120, 105, 115, 100, 95, 90, 85
                    ],
                    'durations': [
                        0.35, 0.25, 0.3, 0.28, 0.4, 0.3, 0.25, 0.35, 0.22, 0.3, 0.24, 0.4, 0.45, 0.32, 0.38
                    ]
                }
                midi_port = mido.open_input(config['device_in']) #type:ignore
                print(f"MIDI mode actif avec le port : {config['device_in']}")
                
                while not stop_event.is_set():
                    
                    for msg in midi_port.iter_pending():
                    # Utiliser polling avec timeout pour vérifier stop_event
                        if msg.type == 'note_on' and msg.velocity > 0:                                                        
                            handle_keydown_midi(msg.note, msg.velocity, state, config, synth, history, last_times, log_callback)
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            handle_keyup_midi(msg.note, state, synth, history, last_times)
                        else:
                            # Petite pause si aucun message MIDI
                            threading.Event().wait(0.01)  # 10ms
                        """
                n =  0
                while n < 15:
                    for note, velocity, duration in zip(default_gesture['note_indices'],
                                                default_gesture['velocities'],
                                                default_gesture['durations']):
                        # Simuler note_on
                        handle_keydown_midi(note, velocity, state, config, synth, history, last_times, log_callback)
                        sleep(duration)  
                        # Simuler note_off
                        handle_keyup_midi(note, state, synth, history, last_times)
                        n+=1
                        """
                midi_port.close()
                
            except (OSError, IOError) as e:
                if log_callback:
                    log_callback(f"Erreur d'ouverture du port MIDI : {config['device_in']} - {e}")
                print(f"Erreur d'ouverture du port MIDI : {config['device_in']} - {e}")

    except Exception as e:
        if log_callback:
            log_callback(f"Erreur dans la boucle d'improvisation: {e}")
        print(f"Erreur dans la boucle d'improvisation: {e}")
    
    finally:
        # Nettoyage final
        if config['mode'] == 'accompagnement' and state.get('note_entries'):
            save_accomp_entries_to_file(state)
            if log_callback:
                log_callback(f"Saved {len(state['note_entries'])} accompaniment entries")
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
