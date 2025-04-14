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
    print(f"SoundFont chargée avec ID: {sfid}")
    fs.program_select(0, sfid, 0, 0)  # Canal 0, SoundFont, Banque 0, Preset 0
    
    # Liste des touches
    key_to_note = [
        pygame.K_a,
        pygame.K_z,
        pygame.K_e,
        pygame.K_r,
        pygame.K_t,
        pygame.K_y,
        pygame.K_u,
        pygame.K_i,
    ]
    
    # Dictionnaires pour mesurer le temps d'appui et mémoriser la note générée
    key_start_time = {}
    note_buffer = {}
    
    # Interface graphique simple
    font = pygame.font.Font(None, 36)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()
    
    # Variables pour afficher l'historique des notes
    note_history = []
    max_history = 10
    
    run = True
    while run:
        screen.fill((240, 240, 240))
        
        # Afficher les instructions
        instructions = font.render("Appuyez sur A-Z-E-R-T-Y-U-I pour jouer des notes", True, (0, 0, 0))
        screen.blit(instructions, (20, 20))
        
        # Afficher l'historique des notes
        history_title = font.render("Historique des notes:", True, (0, 0, 0))
        screen.blit(history_title, (20, 60))
        
        for i, note_info in enumerate(note_history[-max_history:]):
            note_text = font.render(note_info, True, (0, 0, 100))
            screen.blit(note_text, (20, 100 + i * 30))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                if event.key in key_to_note and event.key not in key_start_time:
                    key_start_time[event.key] = time()
                    
                    # Génération immédiate de la note avec une durée par défaut (0)
                    new_state, note = generate_note(previous_state, 0, transitions, supply, midSymbols, p=0.8)
                    
                    # Jouer la note directement via FluidSynth
                    fs.noteon(0, note[0], note[2])
                    
                    note_info = f"KeyDown - Note: {note[0]}, Vel: {note[2]}, State: {new_state}"
                    print(note_info)
                    note_history.append(note_info)
                    
                    note_buffer[event.key] = (new_state, note[0])
            
            elif event.type == pygame.KEYUP:
                if event.key in key_to_note and event.key in key_start_time:
                    duration = time() - key_start_time[event.key]
                    if event.key in note_buffer:
                        state, pitch = note_buffer[event.key]
                        
                        # Mise à jour éventuelle de la note avec la durée mesurée
                        updated_note = (pitch, duration, midSymbols[state][2] if state < len(midSymbols) else 64)
                        previous_state = state
                        
                        # Arrêter la note via FluidSynth
                        fs.noteoff(0, pitch)
                        
                        note_info = f"KeyUp - Note: {pitch}, Dur: {duration:.2f}, State: {state}"
                        print(note_info)
                        note_history.append(note_info)
                        
                        del note_buffer[event.key]
                        del key_start_time[event.key]
        
        pygame.display.flip()
        clock.tick(30)
    
    # Nettoyer
    fs.delete()
    pygame.quit()

if __name__ == '__main__':
    # Pipeline de génération des symboles à partir d'un fichier MIDI.
    midFile = '/home/sylogue/Documents/MuseScore4/Scores/Thirty_Caprices_No._3.mid'
    midFeatures = extract_features(midFile, "polars")
    midSymbols = create_symbole(midFeatures)   # Liste de tuples (pitch, duration, velocity)
    transitions, supply = fo.oracle(sequence=midSymbols)
    previous_state = 0
    get_input(transitions, supply, midSymbols)
