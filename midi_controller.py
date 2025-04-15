import pygame
import pygame.midi
from time import time
from impro import generate_note
import factor_oracle as fo
from create_symbols import extract_features, create_symbole
import mido
import fluidsynth

def get_input(transitions, supply, midSymbols):
    global previous_state
    previous_state = 0


    # Initialisation de pygame pour la gestion des événements clavier.
    pygame.init()
    pygame.display.set_mode((200,200))
    pygame.display.set_mode((800, 400), pygame.RESIZABLE)
    pygame.display.set_caption("FluidSynth MIDI Controller")
    
    fs = fluidsynth.Synth()
    fs.start(driver="alsa")
    
    # Charger la SoundFont
    sfid = fs.sfload("/home/sylogue/stage/Roland_SC-88.sf2")
    fs.program_select(0, sfid, 0, 0)  # Canal 0, SoundFont, Banque 0, Preset 0
    
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

    # Dictionnaires pour mesurer le temps d'appui et mémoriser la note générée
    key_start_time = {}
    note_buffer = {}
    
    # Interface graphique simple
    font = pygame.font.Font(None, 36)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()
    
    melodic_contour = {}
    note_history = {}  # Pour l'affichage
    note_order = 0

    # Variables pour le calcul du gap entre les notes
    last_keyboard_index = None
    
    run = True
    while run:
        screen.fill((240, 240, 240))
        instructions = font.render("Appuyez sur A, Z, E, R, T, Y, U, I pour jouer", True, (0, 0, 0))
        screen.blit(instructions, (20, 20))
        history_title = font.render("Historique des notes:", True, (0, 0, 0))
        screen.blit(history_title, (20, 60))
        
        # Affichage de l'historique (dans l'ordre croissant)
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
                if event.key in key_to_note and event.key not in key_start_time:
                    key_start_time[event.key] = time()
                    current_kb_index = keyboard_mapping[event.key]
                    if last_keyboard_index is None:
                        desired_gap = 100  # Pas de gap pour la première note
                    else:
                        # Par exemple, chaque pas dans le mapping correspond à 2 unités de pitch
                        desired_gap = (current_kb_index - last_keyboard_index) * 10
                    last_keyboard_index = current_kb_index
                    
                    # Génération de la note en passant desired_gap
                    new_state, note = generate_note(previous_state, 0, transitions, supply, midSymbols, desired_gap=desired_gap, p=0.5)
                    
                    # Jouer la note via FluidSynth
                    fs.noteon(0, note[0], note[2])
                    note_info = f"KeyDown - {pygame.key.name(event.key)}: Pitch {note[0]}, Vel {note[2]}, Etat {new_state}"
                    print(note_info)
                    melodic_contour[note_order] = {"keyboard_index": current_kb_index, "pitch": note[0]}
                    note_history[note_order] = note_info
                    note_order += 1
                    
                    note_buffer[event.key] = (new_state, note[0])
            
            elif event.type == pygame.KEYUP:
                if event.key in key_to_note and event.key in key_start_time:
                    duration = time() - key_start_time[event.key]
                    if event.key in note_buffer:
                        state, pitch = note_buffer[event.key]
                        updated_note = (pitch, duration, midSymbols[state][2] if state < len(midSymbols) else 64)
                        previous_state = state
                        fs.noteoff(0, pitch)
                        note_info = f"KeyUp - {pygame.key.name(event.key)}: Pitch {pitch}, Dur {duration:.2f}, Etat {state}"
                        print(note_info)
                        melodic_contour[note_order] = {"keyboard_index": keyboard_mapping[event.key], "pitch": pitch}
                        note_history[note_order] = note_info
                        note_order += 1
                        del note_buffer[event.key]
                    del key_start_time[event.key]
        
        pygame.display.flip()
    
    # Nettoyer
    fs.delete()
    pygame.quit()

if __name__ == '__main__':
    # Pipeline de génération des symboles à partir d'un fichier MIDI.
    midFile = '/home/sylogue/Documents/MuseScore4/Scores/Thirty_Caprices_No._3.mid'
    midFeatures = extract_features(midFile, "polars")
    midSymbols = create_symbole(midFeatures)   # Liste de tuples (pitch, duration, velocity)
    transitions, supply = fo.oracle(sequence=midSymbols)
    get_input(transitions, supply, midSymbols)
