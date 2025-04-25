import pygame
import pygame.midi

from time import time
from impro import generate_note_oracle, generate_note_markov
import factor_oracle as fo
from create_symbols import extract_features, create_symbole
import mido
import fluidsynth
from markov import transition_matrix

def get_input(transitions_oracle, supply, midSymbols, transitions_markov, mode="oracle"):
    global previous_state
    previous_state = 0
    default_velocity = 64
    previous_pitch = midSymbols[0][0] if midSymbols else 60
    lenSymbols = len(midSymbols)
    # Initialisation de pygame pour la gestion des événements clavier.
    pygame.init()
    pygame.display.set_mode((200,200))
    pygame.display.set_mode((800, 400), pygame.RESIZABLE)
    pygame.display.set_caption("FluidSynth MIDI Controller")
    
    fs = fluidsynth.Synth()
    fs.start(driver="alsa")
    
    # Charger la SoundFont
    sfid = fs.sfload("/home/sylogue/stage/Roland_SC-88.sf2")
    fs.program_select(0, sfid, 0, 50)  # Canal 0, SoundFont, Banque 0, Preset au choix
    
    # Liste des touches et définition d'un mapping clavier pour définir un indice pour le contour mélodique
    keyboard_mapping = {
    pygame.K_a: 0,
    pygame.K_z: 1,
    pygame.K_e: 2,
    pygame.K_r: 3,
    pygame.K_t: 4,
    pygame.K_y: 5,
    pygame.K_u: 6,
    pygame.K_i: 7,
    }

    # Liste des touches utilisées (les clés du mapping)
    key_to_note = list(keyboard_mapping.keys())
    note_to_key = list(keyboard_mapping.values())

    # Dictionnaires pour mesurer le temps d'appui et mémoriser la note générée
    key_start_time = {}
    note_buffer = {}
    
    # Interface graphique simple
    font = pygame.font.Font(None, 36)
    screen = pygame.display.get_surface()
    note_history = {}  # Pour l'affichage
    note_order = 0

    # Pour le calcul de la durée effective :
    last_note_end_time = None
    last_note_duration = 0.1  # Valeur par défaut pour la première note
    # juste avant la boucle :
    previous_key_index = None

    run = True
    while run:
        screen.fill((240, 240, 240))
        instructions = font.render("Appuyez sur A, Z, E, R, T, Y, U, I pour jouer", True, (0, 0, 0))
        screen.blit(instructions, (20, 20))
        history_title = font.render("Historique des notes:", True, (0, 0, 0))
        screen.blit(history_title, (20, 60))
        
        # Affichage de l'historique (les 10 dernières entrées)
        y_offset = 100
        for order in sorted(note_history.keys())[-10:]:
            info = note_history[order]
            txt = font.render(f"{order}: {info}", True, (0, 0, 100))
            screen.blit(txt, (20, y_offset))
            y_offset += 30
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                # Si la touche n'est pas déjà appuyée
                if event.key in key_to_note and event.key not in key_start_time:
                    key_start_time[event.key] = time()

                    #calcul du gap
                    current_index = keyboard_mapping[event.key]
                    if previous_key_index is None:
                        gap = 0
                    else:
                        gap = current_index - previous_key_index
                    previous_key_index = current_index

                    # Calcul de la durée effective
                    current_time = time()
                    if last_note_end_time is not None:
                        if current_time >= last_note_end_time:
                            silence = current_time - last_note_end_time
                            duration_eff = last_note_duration + silence
                        else:
                            duration_eff = last_note_duration
                    else:
                        duration_eff = last_note_duration
                    
                    # On prépare une chaîne pour afficher la durée du silence
                    silence_info = ""
                    if last_note_end_time is not None and current_time >= last_note_end_time:
                        silence_info = f" | Silence: {silence:.2f} sec"
                    
                    # Choix du mode d'improvisation
                    if mode == 'oracle':
                        new_state, note = generate_note_oracle(
                            previous_state, duration_eff,
                            transitions_oracle, supply, midSymbols, gap,p=0.2
                        )
                        previous_state = new_state

                    elif mode == 'markov':
                        next_pitch = generate_note_markov(previous_pitch, transitions_markov, notes, gap)
                        note = (next_pitch, duration_eff, default_velocity)
                        previous_pitch = next_pitch
                        new_state = previous_pitch

                    #activation son de la note
 
                    fs.noteon(0, note[0], note[2])
                    if mode =="oracle":
                        note_info = f"KeyDown - {pygame.key.name(event.key)} : Pitch {note[0]}, Vel {note[2]}, État {new_state}/{lenSymbols}{silence_info}, gap {gap}"
                        print(note_info)
                    elif mode =="markov":
                        note_info = f"KeyDown - {pygame.key.name(event.key)} : Pitch {note[0]}, Vel {note[2]}{silence_info}, gap {gap}"
                        print(note_info)
                    note_history[note_order] = note_info
                    note_order += 1
                    note_buffer[event.key] = (new_state, note[0])
            
            #process lorsqu'on relève le doigt du clavier
            elif event.type == pygame.KEYUP:
                if event.key in key_to_note and event.key in key_start_time:
                    duration = time() - key_start_time[event.key]
                    if event.key in note_buffer:
                        state, pitch = note_buffer[event.key]
                        fs.noteoff(0, pitch)
                        if mode =="oracle":
                            note_info = f"KeyUp - {pygame.key.name(event.key)} : Pitch {pitch}, Dur {duration:.2f}, État {state}/{lenSymbols}"
                            print(note_info)
                        elif mode  =="markov":
                            note_info = f"KeyUp - {pygame.key.name(event.key)} : Pitch {pitch}, Dur {duration:.2f}"
                            print(note_info)
                        note_history[note_order] = note_info
                        note_order += 1
                        del note_buffer[event.key]
                    del key_start_time[event.key]
                    
                    last_note_end_time = time()
                    last_note_duration = duration
                    
        pygame.display.flip()
    
    fs.delete()
    pygame.quit()

if __name__ == '__main__':
    # Pipeline de génération des symboles à partir d'un fichier MIDI.
    midFile = '/home/sylogue/Documents/MuseScore4/Scores/Thirty_Caprices_No._3.mid'
    midFeatures = extract_features(midFile, "polars")
    midSymbols = create_symbole(midFeatures)   # Liste de tuples (pitch, duration, velocity)
    transitions_oracle, supply = fo.oracle(sequence=midSymbols)
    transitions_markov, notes = transition_matrix(midSymbols)
    get_input(transitions_oracle, supply, midSymbols,transitions_markov, mode="oracle")
