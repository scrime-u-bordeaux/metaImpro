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
    if event.key in KEYBOARD_MAPPING:
        idx = KEYBOARD_MAPPING[event.key]
        is_black = False
    elif event.key in BLACK_KEY_MAPPING:
        idx = BLACK_KEY_MAPPING[event.key]
        is_black = True
    else:
        # key not bound to any mapping -> ignore
        return 
    
    #scale_filter 
    use_scale_filter = not is_black

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
            use_scale_filter= use_scale_filter
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
        list[mido.Message] : liste de messages MIDI de type note_on
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
            use_scale_filter=is_white_note(note_index),
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
    out_msgs = []

    if pitches_to_play is not None:
        for p in pitches_to_play:
            out_msgs.append(mido.Message('note_on', note=p, velocity=vel, channel=0))
            
    # Store pitches in note_buffer using note_index as key
    state['note_buffer'][note_index] = pitches_to_play
    
    # Initialize key_start dict if it doesn't exist
    if 'key_start' not in last_times:
        last_times['key_start'] = {}
    last_times['key_start'][note_index] = now
    
    log(f"KD MIDI note {note_index} -> pitch {pitches_to_play}, vel {vel}, dur_eff {dur_eff}, gap {gap}")

    return out_msgs

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
        list[mido.Message] : liste de messages MIDI de type note_off
    """
    # Vérifier si la note était en cours
    start_time = last_times['key_start'].get(note_index)
    if start_time is None:
        return [] # Ignorer si aucune note_on correspondante

    # Durée réelle
    dur = time() - start_time
    # Récupérer le pitch et arrêter le son
    pitches = state['note_buffer'].pop(note_index, None)
    out_msgs = []

    for p in pitches:
        out_msgs.append(mido.Message('note_off', note=p, velocity=0, channel=0))
    info = f"KU MIDI note {note_index} -> pitch {pitches}, dur {dur:.2f}"
    log(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][note_index]

    return out_msgs

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
            'note_buffer':   {}
        }

        synth = init_audio(config['sf2_path'], config['audio_driver'])
        random_preset = [0, 11, 12, 16, 18]
        synth_accomp = init_audio(
            config['sf2_path'],
            config['audio_driver'],
            preset=random.choice(random_preset)
        )

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
                midi_in_port_name = config['device_in']
                midi_in_port = mido.open_input(midi_in_port_name) #type:ignore
                print(f"MIDI mode actif avec le port d'entrée : {midi_in_port_name}")

                midi_out_enabled = True
                midi_out_port_name = config['device_out']
                if midi_out_port_name == 'None':
                    midi_out_enabled = False
                elif midi_in_port_name == midi_out_port_name:
                    print('Boucle MIDI détectée, sortie MIDI désactivée')
                    midi_out_enabled = False
                else:
                    midi_out_port = mido.open_output(midi_out_port_name)

                while not stop_event.is_set():
                    sleep(0.01) # is this useful ?

                    generated_midi_events = []

                     # iter_pending was necessary
                     # without it no way to break out from the loop !
                    for msg in midi_in_port.iter_pending():

                        if msg.type == 'note_on' and msg.velocity > 0:                                                        
                            generated_midi_events = handle_keydown_midi(
                                msg.note, msg.velocity,
                                state,
                                config,
                                synth,
                                history,
                                last_times,
                                log_callback,
                            )
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            generated_midi_events = handle_keyup_midi(
                                msg.note,
                                state,
                                synth,
                                history,
                                last_times,
                            )

                    for msg in generated_midi_events:
                        if midi_out_enabled:
                            midi_out_port.send(msg)
                        if config['sf_enable']:
                            if msg.type == 'note_on':
                                synth.noteon(0, msg.note, msg.velocity)
                            elif msg.type == 'note_off':
                                synth.noteoff(0, msg.note)

                print('closing MIDI ports')
                try:
                    midi_in_port.close()
                    midi_out_port.close()
                except:
                    # some port couldn't be closed, much probably because they
                    # were never open, so continue silently
                    pass
                
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
