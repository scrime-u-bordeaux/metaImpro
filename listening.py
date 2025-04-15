import pygame
import time

def play_midi(file_path):
    """
    Plays a MIDI file using pygame.

    Args:
        file_path: The path to the MIDI file.
    """
    # Initialize the pygame mixer
    pygame.mixer.init()

    # Load the MIDI file
    pygame.mixer.music.load(file_path)

    # Start playing the MIDI file
    pygame.mixer.music.play()

    # Keep the program running while the music plays
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    # Clean up pygame
    pygame.mixer.quit()

if __name__ == "__main__":
    # The path to the MIDI file
    output_path ="/home/sylogue/stage/FO/output.mid"
    midi_file_path = '/home/sylogue/Documents/MuseScore4/Scores/Thirty_Caprices_No._3.mid'
    play_midi(midi_file_path)
