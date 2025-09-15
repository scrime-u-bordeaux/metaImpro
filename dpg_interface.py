import mido
import dearpygui.dearpygui as dpg
from dpg_impro import run_impro, stop_impro_thread
import os
import ast
import json
import re
import random
import fluidsynth

model_list = ['oracle', 'markov', 'random', 'accompagnement', 'Autoencoder']
is_impro_running = False
CORPUS_FOLDER = 'corpus'
BOOL_MAP = {"True": True, "False": False} 
EVAL_P_DIR = "eval/probs"
EVAL_G_DIR = "eval/graph"
EVAL_H_DIR = "eval/histogram"
BASENAME = "probs"
EXT = ".json"

# Variables globales pour gérer le thread d'impro
note_history = []
prob_history = []

def get_input_devices():
    return mido.get_input_names() #type:ignore

def get_output_devices():
    return mido.get_output_names()  #type:ignore

def get_corpus():
    """
    Récupère la liste des fichiers MIDI dans le dossier corpus.

    Returns:
        list[str]: Noms des fichiers MIDI disponibles.
    """
    try:
        files = os.listdir(CORPUS_FOLDER)
    except FileNotFoundError:
        return []
    return [f for f in files if f.lower().endswith('.mid') or f.lower().endswith('.midi') or f.lower().endswith('.json')]

def get_pt_files(folder="piano_genie"):
    """Récupère la liste des fichiers .pt dans le dossier piano_genie."""
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return []
    return [f for f in files if f.lower().endswith('.pt')]

def append_log_entry(msg: str):
    global note_history
    global prob_history

     # Message spécial pour la barre de progression
    if msg.startswith("__progress__"):
        try:
            _, state, total = msg.split(":")
            update_oracle_progress(int(state), int(total))
        except:
            pass
        return
    
    if msg.startswith("__markov_probs__"):
        try:
            _, chosen_pitch_str, probs_str, next_prob = msg.split(":", 3)
            chosen_pitch = ast.literal_eval(chosen_pitch_str)
            top_probs = eval(probs_str)  # ex: [(60, 0.5), (62, 0.5)]
            next_prob = float(next_prob)
            update_pie_chart(top_probs, chosen_pitch, next_prob)
            prob_history.append([chosen_pitch,next_prob])

        except Exception as e:
            print(f"Erreur parsing bar chart: {e}")
        return
    
    note_history = note_history[-10:]  # garde seulement les 10 derniers si nécessaire
    note_history.append(msg)
    dpg.configure_item("note_log", items=note_history)


def update_oracle_progress(current_state, total_states):
    """
    Met à jour dynamiquement la barre pour afficher l’état courant.
    """
    if total_states > 1:
        progress = current_state / (total_states - 1)
        dpg.set_value("oracle_progress", progress)

def midi_to_name(midi_pitch):
    m = int(midi_pitch)
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name = names[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"


def make_colors_for_labels(labels, forced_label, forced_color):
    colors = []
    for lab in labels:
        if lab == forced_label:
            colors.append(forced_color)
        else:
            # Générer une nuance de rouge aléatoire
            # Rouge dominant avec variations sur les autres composantes
            red = random.randint(180, 255)    # Rouge fort
            green = random.randint(0, 80)     # Vert faible
            blue = random.randint(0, 80)      # Bleu faible
            colors.append([red, green, blue, 255])
    return colors


def update_pie_chart(top_probs, chosen_pitch, next_prob, bar_tag="markov_pie_series", chosen_tag="chosen_pie"):
    """
    Enhanced version of your update_pie_chart function with better info display
    """
    if not top_probs and not next_prob:
        return
    
    # If the chosen_pitch n'est pas déjà dans top_probs, ajoute-le
    if next_prob is not None:
        present = any(pitch == chosen_pitch for pitch, *_ in top_probs)
        if not present:
            top_probs = top_probs + [(chosen_pitch, next_prob)]
    
    probs = [p[1] for p in top_probs]  # probabilités
    pitches = [p[0] for p in top_probs]  # numéros de pitch
    pitch_names = [midi_to_name(p) for p in pitches]
    
    # Create labels avec pitch + la proba de la note
    pitch_labels = [f"{name} : {prob:.2f}" for name, prob in zip(pitch_names, probs)]
    
    # Trouver le label correspondant au chosen_pitch pour la colormap
    chosen_label = None
    for i, pitch in enumerate(pitches):
        if pitch == chosen_pitch:
            chosen_label = pitch_labels[i]
            break
    
    try:
        # Supprimer l'ancienne colormap si elle existe
        if dpg.does_item_exist("my_cmap"):
            dpg.delete_item("my_cmap")
        
        # Créer la nouvelle colormap
        cmap_colors = make_colors_for_labels(pitch_labels, chosen_label, [0, 180, 0, 255])
        with dpg.colormap_registry():
            cmap_tag = dpg.add_colormap(cmap_colors, True, tag="my_cmap")
        
        # Bind la colormap au plot
        dpg.bind_colormap("markov_plot", "my_cmap")
        
        y_axis_id = "y_axis_markov"
        if dpg.does_item_exist(bar_tag):
            dpg.delete_item(bar_tag)
        if dpg.does_item_exist(y_axis_id):
            dpg.delete_item(y_axis_id)
        
        with dpg.plot_axis(dpg.mvYAxis, parent="markov_plot", no_gridlines=True,
                          no_tick_marks=True, no_tick_labels=True, tag=y_axis_id):
            dpg.set_axis_limits(y_axis_id, 0, 1)
            dpg.add_pie_series(
                0.5, 0.5,  # centre x, y
                0.4,       # rayon
                probs,     # valeurs (proportions)
                pitch_labels,  # étiquettes
                tag=bar_tag,   # identifiant pour référence future
                parent=y_axis_id,
                normalize=True
            )
        # Sort all notes by probability (descending)
        sorted_data = sorted(zip(pitches, probs, pitch_names), key=lambda x: x[1], reverse=True)
        info_lines = ["Notes (probabilité décroissante):\n"]
        info_chosen = []
        
        for pitch, prob, note_name in sorted_data:
            if pitch == chosen_pitch:
                # Highlight chosen note
                info_chosen.append(f" {note_name}: {prob:.3f} (CHOISIE)")
            else:
                info_lines.append(f" {note_name}: {prob:.3f}")
        
        display_text = "\n".join(info_lines)
        display_chosen = "".join(info_chosen)
        
        if dpg.does_item_exist("markov_info_chosen"):
            dpg.set_value("markov_info_chosen", display_chosen)
        if dpg.does_item_exist("markov_info_text"):
            dpg.set_value("markov_info_text", display_text)
            
    except Exception as e:
        print(f"Erreur update pie chart: {e}")

def save_prob_history(prob_history, title: str, mode):
    # Sauvegarde uniquement pour le mode markov
    if mode not in ['markov', 'accompagnement']:
        return

    os.makedirs(EVAL_P_DIR, exist_ok=True)

    # Nettoie le titre pour enlever toute extension
    title_clean = os.path.splitext(title)[0]

    # Récupère les indices existants
    existing = os.listdir(EVAL_P_DIR)
    pattern = re.compile(rf"^{BASENAME}_(\d{{3}}){re.escape(EXT)}$")
    indices = [
        int(m.group(1))
        for f in existing
        if (m := pattern.match(f))
    ]
    order = dpg.get_value('markov_order')
    next_idx = max(indices) + 1 if indices else 1
    filename = f"{BASENAME}_{next_idx:03d}_{title_clean}_ordre{order}{EXT}"
    path = os.path.join(EVAL_P_DIR, filename)

    with open(path, "w") as fp:
        json.dump(prob_history, fp, indent=2)

    print(f"Saved {len(prob_history)} probs into {path}")
    return path
      
def on_model_change(sender, app_data, user_data):
    slider_tag, markov_tag, progress_tag, lvl_tag, n_cand_tag, bpm_tag, accomp_mod_tag, backtrack_checkbox_tag = user_data
    if app_data == 'Autoencoder':
        pt_items = get_pt_files("piano_genie")
        dpg.configure_item('corpus_combo_text', default_value='Choisissez les poids')
        dpg.configure_item(
            'corpus_combo',
            items=pt_items,
            default_value=pt_items[0] if pt_items else None,
        )
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.hide_item(lvl_tag)
        dpg.hide_item(n_cand_tag)
        dpg.hide_item(bpm_tag)
        dpg.hide_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.hide_item("markov_text")
        dpg.hide_item(accomp_mod_tag)
        dpg.hide_item(backtrack_checkbox_tag)
        dpg.hide_item(backtrack_checkbox_tag)
    else:
        corpus_items = get_corpus()
        dpg.configure_item('corpus_combo_text', default_value='Fichier MIDI')
        dpg.configure_item(
            'corpus_combo',
            items=corpus_items,
            default_value=corpus_items[0] if corpus_items else None,
        )

    if app_data == 'oracle':
        dpg.show_item(slider_tag)
        dpg.show_item(progress_tag)
        dpg.show_item(lvl_tag)
        dpg.hide_item(markov_tag)
        dpg.show_item("oracle_text")
        dpg.hide_item("markov_plot")
        dpg.hide_item("markov_text")
        dpg.hide_item(n_cand_tag)
        dpg.hide_item(bpm_tag)
        dpg.hide_item(accomp_mod_tag)
        dpg.hide_item(backtrack_checkbox_tag)

    elif app_data == 'markov':
        dpg.hide_item(slider_tag)
        dpg.show_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.show_item(lvl_tag)
        dpg.show_item(n_cand_tag)
        dpg.show_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.show_item("markov_text")
        dpg.hide_item(bpm_tag)
        dpg.hide_item(accomp_mod_tag)
        dpg.hide_item(backtrack_checkbox_tag)

    elif app_data == 'accompagnement':
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.hide_item(lvl_tag)
        dpg.show_item(n_cand_tag)
        dpg.show_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.show_item("markov_text")
        dpg.show_item(bpm_tag)
        dpg.show_item(accomp_mod_tag)
        dpg.show_item(backtrack_checkbox_tag)

    else:
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.hide_item(lvl_tag)
        dpg.hide_item(n_cand_tag)
        dpg.hide_item(bpm_tag)
        dpg.hide_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.hide_item("markov_text")
        dpg.hide_item(accomp_mod_tag)
        dpg.hide_item(backtrack_checkbox_tag)


# callback pour afficher et récupérer les paramètres
def on_launch(sender, app_data):
    global is_impro_running, prob_history
    
    # Si l'improvisation est en cours, l'arrêter
    if is_impro_running:
        stop_impro()
        return
    lignes = []
    model = dpg.get_value('model_combo')
    lignes.append(f"Modèle : {model}")

    if model == 'oracle':
        lignes.append(f"Probabilité p : {dpg.get_value('oracle_slider_p'):.2f}")
        lignes.append(f"Similarity level : {dpg.get_value('similarity_combo')}")
    if model == 'markov':
        lignes.append(f"Ordre Markov : {dpg.get_value('markov_combo')}")
        lignes.append(f"Similarity level : {dpg.get_value('similarity_combo')}")
        lignes.append(f"Nombre de candidats : {dpg.get_value('n_candidat')}")
    if model == 'accompagnement':
        lignes.append(f"Ordre Markov : {dpg.get_value('markov_combo')}")
        lignes.append(f"Similarity level : {dpg.get_value('similarity_combo')}")
        lignes.append(f"Nombre de candidats : {dpg.get_value('n_candidat')}")
        accomp_mode = dpg.get_value('accomp_mode_combo')
        backtrack_mode = dpg.get_value('backtrack_checkbox')

    if model == 'Autoencoder':
        lignes.append(f"Checkpoint : {dpg.get_value('corpus_combo')}")

    lignes.append(f"Device entrée MIDI : {dpg.get_value('device_in_combo')}")
    lignes.append(f"Device sortie MIDI : {dpg.get_value('device_out_combo')}")
    lignes.append(f"Driver audio : {dpg.get_value('audio_combo')}")
    dpg.set_value('summary_text', ", ".join(lignes))

    cfg = {
        'mode': model,
        'device_in': dpg.get_value('device_in_combo'),
        'device_out': dpg.get_value('device_out_combo'),
        'audio_driver': dpg.get_value('audio_combo'),
        'sf2_path': 'Roland_SC-88.sf2',
        'sf_enable': dpg.get_value('soundfont_enable_checkbox'),
        'p': None,
        'markov_order': None,
        'sim_lvl': None,
        'n_candidat': None,
        'corpus': None,
        'bpm' : None,
        'accomp_mod_tag' :None,
        'backtrack_mod_tag': None,
    }

    chosen = dpg.get_value('corpus_combo')
    print(chosen)
    if model == 'oracle':
        cfg['p'] = float(dpg.get_value('oracle_slider_p'))
        cfg['sim_lvl'] = int(dpg.get_value('similarity_combo'))
        cfg['corpus'] = os.path.join(CORPUS_FOLDER, chosen)
    elif model in ['markov', 'accompagnement']:
        cfg['markov_order'] = int(dpg.get_value('markov_combo'))
        cfg['sim_lvl'] = int(dpg.get_value('similarity_combo'))
        cfg['n_candidat'] = int(dpg.get_value('n_candidat'))
        cfg['corpus'] = os.path.join(CORPUS_FOLDER, chosen)
        if model == 'accompagnement':
            bpm = int(dpg.get_value('bpm_input'))
            cfg['bpm'] = bpm
            lignes.append(f"BPM : {bpm}")
            cfg['accomp_mod_tag'] = accomp_mode
            cfg['backtrack_mode'] = backtrack_mode
    elif model == 'random':
        cfg['corpus'] = os.path.join(CORPUS_FOLDER, chosen)
    else:  # Autoencoder
        cfg['corpus'] = os.path.join('piano_genie', chosen)

    is_impro_running = True
    update_button_text()

    save_prob_history(prob_history, chosen, model)
    run_impro(cfg, append_log_entry)

def stop_impro():
    global is_impro_running
    
    # Arrêter le thread d'improvisation
    stop_impro_thread()
    
    is_impro_running = False
    update_button_text()
    
    # Sauvegarder l'historique à l'arrêt
    mode = dpg.get_value('model_combo')
    save_prob_history(prob_history, dpg.get_value('corpus_combo'), mode)
    
    append_log_entry("Improvisation arrêtée")

# Nouvelle fonction pour mettre à jour le texte du bouton
def update_button_text():
    if is_impro_running:
        dpg.configure_item("launch_button", label="Arrêter l'Impro")
    else:
        dpg.configure_item("launch_button", label="Commencer à Improviser")

# Modifiez la fonction on_exit pour gérer l'arrêt propre
def on_exit():
    global is_impro_running
    if is_impro_running:
        stop_impro()
    else:
        mode = dpg.get_value('model_combo')
        save_prob_history(prob_history, dpg.get_value('corpus_combo'), mode)

def get_available_audio_drivers():
    fs = fluidsynth.Synth()
    fs.start()
    all_known_audio_drivers = [
        'pulseaudio',
        'alsa',
        'coreaudio',
        'dart',
        'dsound',
        'file',
        'jack',
        'oboe',
        'opensles',
        'oss',
        'portaudio',
        'sdl3',
        'sndman',
        'wasapi',
        'waveout',
    ]
    available_audio_drivers = []
    for option in all_known_audio_drivers:
        value = fs.get_setting('audio.{0}.device'.format(option))
        if not value is None:
            available_audio_drivers.append(option)
    fs.delete()
    return available_audio_drivers

dpg.create_context()

with dpg.window(
    # label='Sélection du device',
    label='MetaImpro',
    no_collapse=True,
    no_close=True,
    width=1500,
    height=1300
):
    with dpg.group(horizontal=True):
        with dpg.group():
          dpg.add_text("Périphérique d'entrée MIDI")
          # Combo pour les ports
          input_devices = get_input_devices()
          default_input_device = None
          if len(input_devices) > 0:
              default_input_device = input_devices[0]
          dpg.add_combo(
              tag='device_in_combo',
              items=input_devices,
              default_value=default_input_device,
              width=200
          )
        with dpg.group():
          dpg.add_text('Périphérique de sortie MIDI')
          # Combo pour les ports
          output_devices = [ None ]
          output_devices.extend(get_output_devices())
          default_output_device = None
          if len(output_devices) > 0:
              default_output_device = output_devices[0]
          dpg.add_combo(
              tag='device_out_combo',
              items=output_devices,
              default_value=default_output_device,
              width=200
          )
        with dpg.group():
          dpg.add_text('Pilote audio')
          # Combo pour les ports
          audio_drivers = get_available_audio_drivers()
          default_audio_driver = None
          if len(audio_drivers) > 0:
              default_audio_driver = audio_drivers[0]
          dpg.add_combo(
              tag='audio_combo',
              items=audio_drivers,
              default_value=default_audio_driver,
              width=200
          )
        with dpg.group():
          dpg.add_text(default_value='Fichier MIDI', tag='corpus_combo_text')
          # Combo pour choisir les morceaux
          dpg.add_combo(
              tag='corpus_combo',
              items=get_corpus(),
              default_value='lune_1.mid',
              width=200
          )
        with dpg.group():
          dpg.add_text(default_value=' ', tag='soundfont_enable_checkbox_placeholder')
          dpg.add_checkbox(
              tag='soundfont_enable_checkbox',
              label='soundfont interne',
              default_value=True,
          )

    dpg.add_spacer(height=10)

    dpg.add_text('Choisissez un modèle')
    with dpg.group(horizontal=True):
        # Combo du modèle
        dpg.add_combo(
            tag='model_combo',
            items=model_list,
            default_value='oracle',
            width=200,
            callback=on_model_change,
            user_data=('oracle_slider_p', 'markov_combo', 'oracle_progress', 'similarity_combo', 'n_candidat', 'bpm_input', 'accomp_mode_combo', 'backtrack_checkbox')   # on passe le tag du slider qu’on va créer
        )

        # Combo Markov
        dpg.add_combo(
            tag='markov_combo',
            items= ['0', '1', '2', '3'],
            default_value='1',
            label="Choisissez l'Ordre",
            width=200,

        )

        # Slider Oracle
        dpg.add_slider_float(
            tag="oracle_slider_p",
            label="p",
            default_value=0.7,
            min_value=0.0,
            max_value=1.0,
            width=200
        )
        dpg.add_combo(
            tag="similarity_combo",
            label="Similarity level",
            default_value='1',
            items=['1', '2', '3'],
            width=200
        )
        dpg.add_combo(tag='n_candidat',
                    label='Nombre de candidats',
                    items= ['1', '2', '3', '4', '5'],
                    default_value='5',
                    width=200
        )
        
        dpg.add_input_int(
            tag="bpm_input",
            label="bpm",
            default_value=100,
            min_value=1,
            width=200,
        )
        dpg.add_combo(
            tag='accomp_mode_combo',
            label='Accompagnement mode',
            items=['normal', 'by chord type', 'by interval'],
            default_value='normal',
            width=200,
            show=False
        )
        dpg.add_checkbox(
            tag='backtrack_checkbox',
            label='backtrack/soundfont',
            default_value=False,  # False corresponds to unchecked (was 'False' string)
            show=False
        )
        # On cache les sliders au démarrage
        dpg.hide_item('markov_combo')
        dpg.hide_item('n_candidat')
        dpg.hide_item('bpm_input')

    dpg.add_spacer(height=20)

    with dpg.group(horizontal=True):
        dpg.add_button(label='Commencer à Improviser', callback=on_launch, tag="launch_button")
        dpg.add_text("", tag='summary_text')  # widget de résumé à droite

    dpg.add_spacer(height=10)

    dpg.add_text("Historique des notes :")
    dpg.add_listbox(tag="note_log", items=[], width=1000, num_items=10)

    dpg.add_spacer(height=10)

    # Slider progression Oracle
    dpg.add_text("Progression dans l'oracle :", tag="oracle_text")
    dpg.add_progress_bar(tag="oracle_progress",
        default_value=0.0,
        width=800,
        user_data=('oracle_slider_p', 'markov_combo', 'oracle_progress', 'oracle_slider_lvl')
        )
    
    # Camembert Markov
    dpg.add_text("Camembert des pitchs + probas :", tag="markov_text", show=False)
    with dpg.plot(label="Probas Markov", height=400, width=500, show=False, tag="markov_plot"):
        dpg.add_plot_legend()
    
    dpg.add_text(tag="markov_info_text")
    dpg.add_text(tag="markov_info_chosen", color=(255, 255, 0, 255))


# Création fenêtre
dpg.create_viewport(title='MetaImpro', width=1500, height=1200)
dpg.setup_dearpygui()
dpg.set_exit_callback(on_exit)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

