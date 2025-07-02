import mido
import dearpygui.dearpygui as dpg
from dpg_impro import run_impro
import os
import ast
import json
import re

model_list = ['oracle', 'markov', 'random', 'accompagnement', 'Autoencoder']
CORPUS_FOLDER = 'corpus'
BOOL_MAP = {"True": True, "False": False} 
EVAL_P_DIR = "eval/probs"
EVAL_G_DIR = "eval/graph"
EVAL_H_DIR = "eval/histogram"
BASENAME = "probs"
EXT = ".json"

# Variables globales pour gérer le thread d'impro
_impro_thread = None
_stop_event = None
note_history = []
prob_history = []

def get_device():
    return mido.get_input_names() #type:ignore

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


def update_pie_chart(top_probs, chosen_pitch, next_prob, bar_tag="markov_pie_series", chosen_tag="chosen_pie"):
    """
    Met à jour le graphique avec les probabilités des notes sous forme de camembert.
    """
    if not top_probs and not next_prob:
        return

    # Si le chosen_pitch n’est pas déjà dans top_probs, ajoute-le
    if next_prob is not None:
        present = any(pitch == chosen_pitch for pitch, _ in top_probs)
        if not present:
            top_probs = top_probs + [(chosen_pitch, next_prob)]

    probs = [p[1] for p in top_probs]  # probabilités
    pitches = [pitch for pitch, _ in top_probs]  # numéros de pitch

    # on créé des labels avec pitch  + la proba de la note

    pitch_labels = [f"{pitch} : {prob:.2f}" for pitch, prob in top_probs]

    try:
        # Vérifier si l'axe Y existe déjà, sinon le créer
        y_axis_id = "y_axis_markov"

        # Supprimer l'ancienne série de données
        if dpg.does_item_exist(bar_tag):
            dpg.delete_item(bar_tag)

        # Si l'axe Y existe, on le nettoie
        if dpg.does_item_exist(y_axis_id):
            dpg.delete_item(y_axis_id)

        # Créer un nouvel axe Y pour le graphique
        with dpg.plot_axis(dpg.mvYAxis, parent="markov_plot", no_gridlines=True,
                           no_tick_marks=True, no_tick_labels=True, tag=y_axis_id):
            dpg.set_axis_limits(y_axis_id, 0, 1)

            # Création du camembert avec les nouvelles données
            dpg.add_pie_series(
                0.5, 0.5,         # centre x, y
                0.4,              # rayon
                probs,            # valeurs (proportions)
                pitch_labels,     # étiquettes
                tag=bar_tag,      # identifiant pour référence future
                parent=y_axis_id,
                normalize=True
            )
        
        # Mettre à jour le texte descriptif
        if dpg.does_item_exist("markov_info_text"):
            # Trouver l'index de la note choisie dans les probas
            chosen_idx = -1
            for i, pitch in enumerate(pitches):
                if pitch == chosen_pitch:
                    chosen_idx = i
                    break

            if chosen_idx >= 0:
                chosen_prob = probs[chosen_idx]
                chosen_note = pitch_labels[chosen_idx]
                dpg.set_value("markov_info_text", f"Note choisie: {chosen_note} (prob: {chosen_prob:.2f})")
            else:
                dpg.set_value("markov_info_text", f"Note choisie: {chosen_pitch} (non représentée dans le top)")
        
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

    slider_tag, markov_tag, progress_tag, lvl_tag, contour_tag = user_data
    if app_data == 'Autoencoder':
        # replace corpus list with .pt checkpoints
        pt_items = get_pt_files("piano_genie")
        dpg.configure_item('corpus_combo',
                           items=pt_items,
                           default_value=pt_items[0] if pt_items else None,
                           label='Choisissez les poids')
        # hide irrelevant controls
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.hide_item(lvl_tag)
        dpg.hide_item(contour_tag)
        dpg.hide_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.hide_item("markov_text")
    else:
        # restore MIDI/json listing
        corpus_items = get_corpus()
        dpg.configure_item('corpus_combo',
                           items=corpus_items,
                           default_value=corpus_items[0] if corpus_items else None,
                           label="Choisissez un morceau")
    # fonctionnement normal
    if app_data == 'oracle':
        dpg.show_item(slider_tag)
        dpg.show_item(progress_tag)
        dpg.show_item(lvl_tag)
        dpg.show_item(contour_tag)
        dpg.hide_item(markov_tag)
        dpg.show_item("oracle_text")
        dpg.hide_item("markov_plot")
        dpg.hide_item("markov_text")
    elif app_data in ['markov', 'accompagnement']:
        dpg.hide_item(slider_tag)
        dpg.show_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.show_item(lvl_tag)
        dpg.show_item(contour_tag)
        dpg.show_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.show_item("markov_text")
    else:
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)
        dpg.hide_item(progress_tag)
        dpg.hide_item(lvl_tag)
        dpg.hide_item(contour_tag)
        dpg.hide_item("markov_plot")
        dpg.hide_item("oracle_text")
        dpg.hide_item("markov_text")

# callback pour afficher et récupérer les paramètres
def on_launch(sender, app_data):
    lignes = []
    model = dpg.get_value('model_combo')
    lignes.append(f"Modèle : {model}")

    # Paramètres spécifiques
    if model == 'oracle':
        lignes.append(f"Probabilité p : {dpg.get_value('oracle_slider_p'):.2f}")
        lignes.append(f"Similarity level : {dpg.get_value('similarity_combo')}")
    if model in ['markov', 'accompagnement']:
        lignes.append(f"Ordre Markov : {dpg.get_value('markov_combo')}")
        lignes.append(f"Similarity level : {dpg.get_value('similarity_combo')}")
    if model == 'Autoencoder':
        lignes.append(f"Checkpoint : {dpg.get_value('corpus_combo')}")

    # Toujours afficher device
    lignes.append(f"Device MIDI : {dpg.get_value('device_combo')}")

    # Récapitulatif
    dpg.set_value('summary_text', ", ".join(lignes))

    # Construction du config dict
    # On récupère la valeur de corpus_combo, qui sera soit un .mid/.json, soit un .pt selon on_model_change
    chosen = dpg.get_value('corpus_combo')
    # Common fields (toujours présents)
    cfg = {
        'mode': model,
        'device': dpg.get_value('device_combo'),
        'sf2_path': 'Roland_SC-88.sf2',
        # champs partagés (même si None)
        'p': None,
        'markov_order': None,
        'sim_lvl': None,
        'contour': None,
        'corpus': None,
    }

    # Remplissage selon mode
    if model == 'oracle':
        cfg['p']            = float(dpg.get_value('oracle_slider_p'))
        cfg['sim_lvl']      = int(dpg.get_value('similarity_combo'))
        cfg['contour']      = BOOL_MAP[dpg.get_value('contour_combo')]
        cfg['corpus']       = os.path.join(CORPUS_FOLDER, chosen)
    elif model in ['markov', 'accompagnement']:
        cfg['markov_order'] = int(dpg.get_value('markov_combo'))
        cfg['sim_lvl']      = int(dpg.get_value('similarity_combo'))
        cfg['contour']      = BOOL_MAP[dpg.get_value('contour_combo')]
        cfg['corpus']       = os.path.join(CORPUS_FOLDER, chosen)
    elif model == 'random':
        cfg['corpus']       = os.path.join(CORPUS_FOLDER, chosen)
    elif model == 'Autoencoder':
        # On met le chemin du .pt dans corpus pour que load_symbols / improvisation_loop ne plante pas
        cfg['corpus'] = os.path.join('piano_genie', chosen)
        # pas de p, pas de markov, pas de contour, sim_lvl etc.
    else:
        # fallback (ne devrait pas arriver)
        cfg['corpus'] = os.path.join(CORPUS_FOLDER, chosen)

    # Sauvegarde et lancement
    save_prob_history(prob_history, chosen, model)
    run_impro(cfg, append_log_entry)
def on_exit():
    mode = dpg.get_value('model_combo')
    save_prob_history(prob_history, dpg.get_value('corpus_combo'), mode)


dpg.create_context()

with dpg.window(label='Sélection du device', width=1300, height=1300):
    dpg.add_text("Choisissez un device")
    with dpg.group(horizontal=True):

        # Combo pour les ports
        dpg.add_combo(
            tag='device_combo',
            items=get_device(),
            default_value='Midi Through:Midi Through Port-0 14:0',
            width=200
        )
        # Combo pour choisir les morceaux
        dpg.add_combo(
            tag='corpus_combo',
            items=get_corpus(),
            default_value='lune_1.mid',
            label='Choisissez un morceau',
            width=200
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
            user_data=('oracle_slider_p', 'markov_combo', 'oracle_progress', 'similarity_combo', 'contour_combo')   # on passe le tag du slider qu’on va créer
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
        dpg.add_combo(
            tag="contour_combo",
            label="Contour",
            default_value='True',
            items=['True', 'False'],
            width=200
        )
        

        # On cache les sliders au démarrage
        dpg.hide_item('markov_combo')

    dpg.add_spacer(height=20)

    with dpg.group(horizontal=True):
        dpg.add_button(label='Commencer à Improviser', callback=on_launch)
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

        # create x axis
        dpg.add_plot_axis(dpg.mvXAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
        dpg.set_axis_limits(dpg.last_item(), 0, 1)
        
        # Création de l'axe Y initial avec une série vide
        with dpg.plot_axis(dpg.mvYAxis, parent="markov_plot", no_gridlines=True, no_tick_marks=True, no_tick_labels=True):
            dpg.set_axis_limits(dpg.last_item(), 0, 1)
            # Série initiale d'exemple
            dpg.add_pie_series(
                x=1,
                y=1,
                radius=1,
                values=[0.6, 0.3, 0.1],
                labels= ["pitch_1 : probs_1", "pitch_2 : probs_2", "pitch_3 : probs_3"],
            )
    dpg.add_text(tag="markov_info_text")
    

# Création fenêtre
dpg.create_viewport(title='MetaImpro', width=1200, height=950)
dpg.setup_dearpygui()
dpg.set_exit_callback(on_exit)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

