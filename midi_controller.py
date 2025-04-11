import pygame
import mido
from mido import Message
import time

def midi_controller():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Virtual MIDI Keyboard")
    
    # Désactivation de la répétition de touches
    pygame.key.set_repeat(0)
    
    try:
        output = mido.open_output()  # Choisissez le port souhaité
    except IOError as e:
        print("Erreur lors de l'ouverture du port MIDI :", e)
        return

    # On met du piano
    msg_pc = Message('program_change', channel=0, program=0)  # numéro 0 pour piano acoustique
    output.send(msg_pc)

    key_to_note = {
        pygame.K_a: 60,  # Do
        pygame.K_z: 62,  # Ré
        pygame.K_e: 64,  # Mi 
        pygame.K_r: 65,  # Fa
        pygame.K_t: 67,  # Sol
        pygame.K_y: 69,  # La
        pygame.K_u: 71,  # Si
        pygame.K_i: 72,  # Do aigu
    }

    keys_pressed = {}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_note and not keys_pressed.get(event.key, False):
                    note = key_to_note[event.key]
                    # Activer éventuellement le sustain (si désiré)
                    sustain_on = Message('control_change', channel=0, control=64, value=127)
                    output.send(sustain_on)
                    
                    msg = Message('note_on', channel=0, note=note, velocity=64)
                    output.send(msg)
                    keys_pressed[event.key] = True
                    print(f"Touche {pygame.key.name(event.key)} enfoncée, note {note} ON")
            
            elif event.type == pygame.KEYUP:
                if event.key in key_to_note and keys_pressed.get(event.key, False):
                    note = key_to_note[event.key]
                    msg = Message('note_off', channel=0, note=note, velocity=0)
                    output.send(msg)
                    # Désactivation du sustain si activé pour cette touche
                    sustain_off = Message('control_change', channel=0, control=64, value=0)
                    output.send(sustain_off)
                    
                    keys_pressed[event.key] = False
                    print(f"Touche {pygame.key.name(event.key)} relâchée, note {note} OFF")
        
        screen.fill((30, 30, 30))
        pygame.display.flip()

    pygame.quit()
    output.close()

if __name__ == '__main__':
    midi_controller()
