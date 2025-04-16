# Création d'une chaine de Markov d'ordre 1 sur les pitches d'un fichier midi

import numpy as np
from create_symbols import extract_features, create_symbole
import mido

def transition_matrix(midSymbols):

    nb_states = len(midSymbols)
    pitches = [pitch[0] for pitch in midSymbols]
    notes = np.unique(pitches)
    matrix = np.zeros((len(notes), len(notes)))   
    # Dictionnaire pour mapper chaque note à un index
    note_to_index = {note: index for index, note in enumerate(notes)}

    # Comptage des transitions
    for i in range(len(pitches) - 1):
        current_state = note_to_index[pitches[i]]
        next_state = note_to_index[pitches[i + 1]]
        matrix[current_state, next_state] += 1

    # Normalisation de la matrice
    row_sums = matrix.sum(axis=1, keepdims=True)

    transition_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    return transition_matrix, notes

def generate_sequence(notes, transitions_matrix, length=100, start_note=None, duration = 150):

    if start_note is None:
        current_state = np.random.choice(notes)
    else:
        current_state = start_note

    sequence = []

    # Dictionnaires pour mapper chaque note à un index
    note_to_index = {note: index for index, note in enumerate(notes)}
    index_to_note = {index: note for note, index in note_to_index.items()}

    for _ in range(length):
        # Ajouter le pitch actuel et la durée fixe à la séquence
        sequence.append((current_state, duration))

        # Sélectionner le prochain état en fonction des probabilités de transition
        current_index = note_to_index[current_state]
        next_index = np.random.choice(len(notes), p=transitions_matrix[current_index])
        current_state = index_to_note[next_index]

    return sequence

def sequence_to_midi(sequence, output_file='output_markov.mid'):
    """
    Création d'un fichier MIDI à partir de la séquence générée.
    
    Paramètres:
      sequence (list of tuples): Liste de tuples (pitch, duration) de la séquence générée.
      output_file (str): Nom du fichier de sortie.
    """
    # Création d'un nouveau fichier MIDI et d'une piste
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    for pitch, duration in sequence:
        # Note on : time=0 pour démarrer immédiatement après la note précédente
        note_on = mido.Message('note_on', note=pitch, velocity=80, time=0)
        track.append(note_on)
        # Note off : le temps correspond à la durée de la note
        note_off = mido.Message('note_off', note=pitch, velocity=0, time=duration)
        track.append(note_off)
    
    mid.save(output_file)


if __name__ == '__main__':
    # Pipeline de génération des symboles à partir d'un fichier MIDI.
    midFile = '/home/sylogue/Documents/MuseScore4/Scores/Thirty_Caprices_No._3.mid'
    midFeatures = extract_features(midFile, "polars")
    midSymbols = create_symbole(midFeatures) 
    transitions, notes = transition_matrix(midSymbols)
    sequence = generate_sequence(notes, transitions)
    sequence_to_midi(sequence)