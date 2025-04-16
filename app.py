import gradio as gr


def update_slider(model_choice):
    """
    Met à jour la visibilité du slider en fonction du modèle choisi.
    Si l'utilisateur choisit "oracle", le slider devient visible.
    Sinon, il reste caché.
    """
    if model_choice == "oracle":
        # On retourne une mise à jour qui rend le slider visible
        return gr.Slider(visible=True)
    else:
        return gr.Slider(visible=False)

def process(model_choice, probability):
    """
    Traite la sélection de modèle.
    Si le modèle "oracle" est choisi, il utilise la valeur de probabilité.
    """
    if model_choice == "oracle":
        return f"Vous avez choisi l'oracle avec p = {probability}"
    else:
        return f"Vous avez choisi {model_choice}"

with gr.Blocks() as demo:
    # Sélection du modèle via un bouton radio
    model_choice = gr.Radio(choices=["markov", "oracle"], label="Sélectionnez un modèle")
    
    # Slider pour la probabilité, caché par défaut
    probability = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.05, label="p =", visible=False)

    
    # Zone de sortie qui affichera le résultat
    output = gr.Textbox(label="Résultat")
    
    # Déclencheur qui met à jour la visibilité du slider lorsque le modèle est modifié.
    model_choice.change(fn=update_slider, inputs=model_choice, outputs=probability)
    
    # Texte d'instructions
    instructions = gr.Markdown("**Appuyez sur les touches [a, z, e, r, t, y, u, i] pour improviser**")
    
    # Bouton pour lancer le traitement
    submit_btn = gr.Button("Valider")
    submit_btn.click(fn=process, inputs=[model_choice, probability], outputs=output)

demo.launch()
