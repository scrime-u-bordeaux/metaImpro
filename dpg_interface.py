import mido
import threading
import dearpygui.dearpygui as dpg
from dpg_impro import run_impro
import os

model_list = ['oracle', 'markov', 'SuperTransformerDiffuseurLSTM']
CORPUS_FOLDER = 'corpus'

# Variables globales pour gérer le thread d'impro
_impro_thread = None
_stop_event = None
note_history = []

def get_device():
    return mido.get_input_names()

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
    return [f for f in files if f.lower().endswith('.mid')]

def append_log_entry(msg: str):
    global note_history
    note_history = note_history[-10:]  # garde seulement les 9 derniers si nécessaire
    note_history.append(msg)
    dpg.configure_item("note_log", items=note_history)

def on_model_change(sender, app_data, user_data):
    slider_tag, markov_tag =user_data
    if app_data == 'oracle':
        dpg.show_item(slider_tag)
        dpg.hide_item(markov_tag)
    elif app_data == 'markov':
        dpg.hide_item(slider_tag)
        dpg.show_item(markov_tag)
    else:
        dpg.hide_item(slider_tag)
        dpg.hide_item(markov_tag)


# callback pour afficher et récupérer les paramètres
def on_launch(sender, app_data):
    lignes = []
    
    if dpg.is_item_shown('model_combo'):
        val = dpg.get_value('model_combo')
        lignes.append(f"Modèle : {val}")

    if dpg.is_item_shown('oracle_slider'):
        lignes.append(f"Probabilité p : {dpg.get_value('oracle_slider'):.2f}")

    if dpg.is_item_shown('markov_combo'):
        lignes.append(f"Ordre Markov : {dpg.get_value('markov_combo')}")

    lignes.append(f"Device MIDI : {dpg.get_value('device_combo')}")
    lignes.append(f"Morceau : {dpg.get_value('corpus_combo')}")
    recap = ", ".join(lignes)
    dpg.set_value('summary_text', recap)

    mode = dpg.get_value('model_combo')
    device = dpg.get_value('device_combo')
    corpus_file = dpg.get_value('corpus_combo')
    p_value = dpg.get_value('oracle_slider') if dpg.is_item_shown('oracle_slider') else None
    markov_order = dpg.get_value('markov_combo') if dpg.is_item_shown('markov_combo') else 1 #mettre  à un sinon la fonction vlmc_table bug pour créer la table de transition 
    markov_order = int(markov_order) #on cast un int car c'est un str

    # Construction du chemin complet du corpus
    corpus_path = os.path.join(CORPUS_FOLDER, corpus_file)

    config = {
        'mode': mode,
        'device': device,
        'corpus': corpus_path,
        'p': p_value,
        'markov_order': markov_order,
        'sf2_path': 'Roland_SC-88.sf2'
    }

    # Lancement du thread d'impro
    run_impro(config, append_log_entry)



dpg.create_context()

with dpg.window(label='Sélection du device', width=1200, height=400):
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
            user_data=('oracle_slider', 'markov_combo')   # on passe le tag du slider qu’on va créer
        )

        # Combo Markov
        dpg.add_combo(
            tag='markov_combo',
            items= [1, 2, 3],
            default_value=1,
            label="Choisissez l'Ordre",
            width=200,

        )

        # Slider Oracle
        dpg.add_slider_float(
            tag="oracle_slider",
            label="p",
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
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

dpg.create_viewport(title='MetaImpro', width=1200, height=600)
dpg.setup_dearpygui()



dpg.show_viewport()

dpg.start_dearpygui()
dpg.destroy_context()

