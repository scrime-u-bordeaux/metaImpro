from time import time
from impro import generate_note_oracle, generate_note_markov
import factor_oracle as fo
from create_symbols import extract_features, create_symbole
import mido
import fluidsynth
from markov import transition_matrix
import pygame
import threading

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


def load_symbols_from_midi(midi_path: str):
    """
    Extrait les symboles musicaux et calcule les transitions pour Oracle et Markov.

    Args:
        midi_path (str): Chemin vers le fichier MIDI.

    Returns:
        tuple: mid_symbols, transitions_oracle, supply, transitions_markov, notes
    """
    features = extract_features(midi_path)
    mid_symbols = create_symbole(features)
    transitions_oracle, supply = fo.oracle(sequence=mid_symbols)
    transitions_markov, notes = transition_matrix(mid_symbols)
    return mid_symbols, transitions_oracle, supply, transitions_markov, notes


def handle_keydown(event, state, config, synth, history, last_times):
    """
    Gère un événement KEYDOWN pour générer et jouer une note d'improvisation.

    Args:
        event: pygame KEYDOWN event
        state: dict, contient prev_state, prev_pitch, note_buffer, etc.
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

    # Génération de la note
    if config['mode'] == 'oracle':
        new_state, note, links = generate_note_oracle(
            state['prev_state'], dur_eff,
            state['trans_oracle'], state['supply'], state['mid_symbols'], gap,
            p=config.get('p', 0.5), contour=False
        )
        state['prev_state'] = new_state
    else:
        next_pitch, next_prob, top_probs = generate_note_markov(
            state['prev_pitch'], state['trans_markov'], state['notes'], gap, contour=True
        )
        note = (next_pitch, dur_eff, config.get('default_velocity', 64))
        state['prev_pitch'] = next_pitch

    # Jouer la note
    synth.noteon(0, note[0], note[2])
    # Stocker dans note_buffer pour keyup
    state['note_buffer'][event.key] = note[0]

    info = f"KD {pygame.key.name(event.key)} -> pitch {note[0]}, vel {note[2]}, dur_eff {dur_eff:.2f}, gap {gap}"
    history.append(info)


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
    history.append(info)

    # Mettre à jour les temps
    last_times['last_note_end'] = time()
    last_times['last_note_duration'] = dur
    # Retirer le start
    del last_times['key_start'][event.key]



def improvisation_loop(config, stop_event):
    """Boucle principale d'improvisation musicale, capte les événements clavier en mode headless.

    Args:
        config (dict): Configuration contenant 'mode', 'device', 'corpus', 'p', 'markov_order', 'sf2_path'.
        stop_event (threading.Event): Événement pour arrêter la boucle.
    Returns:
        None
    """
    mid_symbols, trans_oracle, supply, trans_markov, notes = load_symbols_from_midi(config['corpus'])
    state = {'prev_state': 0,
             'prev_pitch': mid_symbols[0][0] if mid_symbols else 60,
             'mid_symbols': mid_symbols,
             'trans_oracle': trans_oracle,
             'supply': supply,
             'trans_markov': trans_markov,
             'notes': notes,
             'note_buffer': {}}

    synth = init_audio(config['sf2_path'])
    pygame.init()
    pygame.display.set_mode((1, 1))

    history = []
    last_times = {'key_start': {}, 'last_note_end': None,
                  'last_note_duration': 0.1, 'prev_key_index': None}

    while not stop_event.is_set():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                stop_event.set()
            elif ev.type == pygame.KEYDOWN and ev.key in KEYBOARD_MAPPING:
                last_times['key_start'][ev.key] = time()
                handle_keydown(ev, state, config, synth, history, last_times)
            elif ev.type == pygame.KEYUP and ev.key in last_times['key_start']:
                handle_keyup(ev, state, synth, history, last_times)

    synth.delete()
    pygame.quit()


def run_impro(config):
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
                                     args=(config, _stop_event),
                                     daemon=True)
    _impro_thread.start()
    return _impro_thread
