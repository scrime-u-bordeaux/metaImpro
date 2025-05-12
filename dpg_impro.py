from time import time
from impro import generate_note_oracle
import factor_oracle as fo
from create_symbols import extract_features, create_symbole
import mido
import fluidsynth
from markov import build_vlmc_table, generate_note_vlmc

import pygame
import threading

log = print
# Mapping clavier pour contour mélodique
KEYBOARD_MAPPING = {
    pygame.K_a: 0, pygame.K_z: 1, pygame.K_e: 2, pygame.K_r: 3,
    pygame.K_t: 4, pygame.K_y: 5, pygame.K_u: 6, pygame.K_i: 7,
}

# Variables globales pour gérer le thread d'impro
_impro_thread = None
_stop_event = None

def init_audio(sf2_path: str, driver: str = "pulseaudio", preset: int = 50):
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


def load_symbols_from_midi(midi_path: str, markov_order: int = 1):
    """
    Extrait les symboles musicaux et calcule les transitions pour Oracle et Markov.

    Args:
        midi_path (str): Chemin vers le fichier MIDI.

    Returns:
        tuple: mid_symbols, trans_oracle, supply, vlmc_table, notes
    """
    features = extract_features(midi_path)
    mid_symbols = create_symbole(features)
    trans_oracle, supply = fo.oracle(sequence=mid_symbols)
    vlmc_table = build_vlmc_table(mid_symbols, max_order=markov_order)
    notes = sorted({s[0] for s in mid_symbols})
    return mid_symbols, trans_oracle, supply, vlmc_table, notes


def handle_keydown(event, state, config, synth, history, last_times):
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
        new_state, note, _ = generate_note_oracle(
            state['prev_state'], dur_eff,
            state['trans_oracle'], state['supply'], state['mid_symbols'], gap,
            p=config['p'], contour=False
        )
        state['prev_state'] = new_state
    else:
        # Markov: contexte variable
        next_pitch, next_prob, top_probs = generate_note_vlmc(
            state['context'], state['vlmc_table'], state['notes'],
            gap, contour=True, max_order=config['markov_order']
        )
        # Mise à jour du contexte
        state['context'].append(next_pitch)
        # Conserver uniquement les derniers N selon markov_order
        max_ord = config['markov_order']
        if max_ord > 0 and len(state['context']) > max_ord:
            state['context'] = state['context'][-max_ord:]
        note = (next_pitch, dur_eff, 64)


    # Jouer note
    synth.noteon(0, note[0], note[2])
    state['note_buffer'][event.key] = note[0]
    log(
        f"KD {pygame.key.name(event.key)} -> pitch {note[0]}, vel {note[2]}, dur_eff {dur_eff:.2f}, gap {gap}"
    )


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
    pitch = state['note_buffer'].pop(event.key, None)
    if pitch is not None:
        synth.noteoff(0, pitch)
    info = f"KU {pygame.key.name(event.key)} -> pitch {pitch}, dur {dur:.2f}"
    log(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][event.key]

def handle_keydown_midi(note_index, velocity, state, config, synth, history, last_times):
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
        new_state, note, _ = generate_note_oracle(
            state['prev_state'], dur_eff,
            state['trans_oracle'], state['supply'], state['mid_symbols'], gap,
            p=config['p'], contour=False
        )
        state['prev_state'] = new_state
    else:
        # Markov: contexte variable
        next_pitch, next_prob, top_probs = generate_note_vlmc(
            state['context'], state['vlmc_table'], state['notes'],
            gap, contour=True, max_order=config['markov_order']
        )
        # Mise à jour du contexte
        state['context'].append(next_pitch)
        # Conserver uniquement les derniers N selon markov_order
        max_ord = config['markov_order']
        if max_ord > 0 and len(state['context']) > max_ord:
            state['context'] = state['context'][-max_ord:]
        note = (next_pitch, dur_eff, velocity)


    # Jouer la note
    synth.noteon(0, note[0], note[2])
    state['note_buffer'][note_index] = note[0]
    last_times['key_start'][note_index] = now
    log(
        f"KD MIDI note {note_index} -> pitch {note[0]}, vel {note[2]}, dur_eff {dur_eff:.2f}, gap {gap}"
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
    pitch = state['note_buffer'].pop(note_index, None)
    if pitch is not None:
        synth.noteoff(0, pitch)
    info = f"KU MIDI note {note_index} -> pitch {pitch}, dur {dur:.2f}"
    log(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][note_index]

def midi_listener(config, state, synth, history, last_times):
    """
    Gère les note_on et note_off d'un clavier midi

    Args:
        config (dict): Configuration contenant 'mode', 'device', 'corpus', 'p', 'markov_order', 'sf2_path'.
        stop_event (threading.Event): Événement pour arrêter la boucle.
    Returns:
        None
    """
    port = config['device']
    for msg in port:
        if msg.type == 'note_on' and msg.velocity > 0:
            note_index = msg.note
            handle_keydown_midi(note_index, msg.velocity, state, config, synth, history, last_times)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            note_index = msg.note
            handle_keyup_midi(note_index, state, synth, history, last_times)
    
def improvisation_loop(config, stop_event, log_callback=None):

    history = []

    global log
    def log(msg):
        if log_callback:
            log_callback(msg)
        history.append(msg)


    mid_symbols, trans_oracle, supply, vlmc_table, notes = load_symbols_from_midi(
        config['corpus'], markov_order=config['markov_order']
    )
    state = {
        'prev_state': 0,
        'context': [mid_symbols[0][0] if mid_symbols else 60],
        'mid_symbols': mid_symbols,
        'trans_oracle': trans_oracle,
        'supply': supply,
        'vlmc_table': vlmc_table,
        'notes': notes,
        'note_buffer': {}
    }

    synth = init_audio(config['sf2_path'])
    
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
                    handle_keydown(ev, state, config, synth, history, last_times)
                elif ev.type == pygame.KEYUP and ev.key in last_times['key_start']:
                    handle_keyup(ev, state, synth, history, last_times)
        pygame.quit()

    else:
        try:
            midi_port = mido.open_input(config['device'])
            print(f"✅ MIDI mode actif avec le port : {config['device']}")
            for msg in midi_port:
                if stop_event.is_set():
                    break
                if msg.type == 'note_on' and msg.velocity > 0:
                    handle_keydown_midi(msg.note, msg.velocity, state, config, synth, history, last_times)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    handle_keyup_midi(msg.note, state, synth, history, last_times)
            midi_port.close()
        except (OSError, IOError):
            print(f"❗ Erreur d'ouverture du port MIDI : {config['device']}")

        pygame.quit()

    synth.delete()


def run_impro(config, log_callback=None):
    """Lance (ou relance) la boucle d'improvisation dans un thread daemon.

    Args:
        config (dict): Configuration d'improvisation.
    Returns:
        threading.Thread: Le thread en cours d'exécution (daemon).
    """
    global _impro_thread, _stop_event
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
