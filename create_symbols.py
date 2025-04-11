import mido
from mido import Message
import os
import json
from tqdm import tqdm
import time

def extract_features(midi_file):
    """
    Récupération des paramètres déclarés dans le constructeur
    Args:
        midi_file: Chemin vers le fichier MIDI
    Returns:
        Dictionary contenant les séquences extraites avec seulement duration, pitch, chord et velocity
    """
    try:
        # Création d'un objet MidiFile pour agire sur le midi_file
        mid = mido.MidiFile(midi_file)
        # Préparation pour l'extraction de notes
        notes = []
        active_notes = {} # {(track, channel, note): start_time}

        # Analyser chaque piste et message
        for track_index, track in enumerate(mid.tracks):
            # tenir compte du temps dans la track
            current_time = 0
            for msg in track:
                # Mettre à jour le temps courant
                current_time += msg.time

                # Ignorer tous les messages liés au tempo
                if msg.type in ['set_tempo', 'time_signature']:
                    continue

                # Traitement des note_on, note_off
                # Note_on activation
                if msg.type == 'note_on' and msg.velocity > 0:
                    # clé unique qui identifie la piste, le canal Midi ainsi que la note
                    key = (track_index, msg.channel, msg.note)
                    # la clé permet d'accéder aux valeurs du dictionnaires qui sont le temps actuel du midi_file et sa vélocitée
                    active_notes[key] = {
                        'start_time': current_time,
                        'velocity': msg.velocity
                    }
                # Note_off activation (Note_off ou velocité =0)
                elif msg.type == "note_off" or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (track_index, msg.channel, msg.note)
                    if key in active_notes:
                        # on récupère le start_time de la note_off
                        start_info = active_notes.pop(key)
                        start_time = start_info["start_time"]
                        # on calcule la durée d'une note en ticks
                        duration_ticks = current_time - start_time
                        # on créé un symbole de note
                        notes.append({
                            'pitch': msg.note,
                            'start_ticks': start_time,
                            'duration_ticks': duration_ticks,
                            'velocity': start_info['velocity']
                        })

        # Trier les notes par temps de début
        notes = sorted(notes, key=lambda x: x['start_ticks'])

        # Identifier les notes qui font partie d'un accord (basé uniquement sur la proximité temporelle en ticks)
        # Indépendamment du tempo qui peut varier (rubato, changements dans la partition)
        CHORD_THRESHOLD_TICKS = 20  # Seuil fixe en ticks pour considérer des notes comme faisant partie d'un accord
        
        # Grouper les notes en accords
        chords = []
        current_chord = []
        
        for i, note in enumerate(notes):
            if not current_chord:
                # Premier accord
                current_chord.append(note)
            elif abs(note['start_ticks'] - current_chord[0]['start_ticks']) <= CHORD_THRESHOLD_TICKS:
                # La note fait partie de l'accord courant
                current_chord.append(note)
            else:
                # La note commence un nouvel accord
                chords.append(current_chord)
                current_chord = [note]
        
        # Ajouter le dernier accord s'il existe
        if current_chord:
            chords.append(current_chord)
        
        # Marquer chaque note comme faisant partie d'un accord ou non
        chord_ids = {}
        for i, chord in enumerate(chords):
            for note in chord:
                note['chord_id'] = i
                note['in_chord'] = len(chord) > 1

        # Création de symboles qui ne contiennent que les informations demandées
        symbols = []
        for i, note in enumerate(notes):
            # Créer un symbole pour chaque note qui inclut uniquement les informations requises
            symbol = {
                'id': i,
                'pitch': note['pitch'],
                'duration': note['duration_ticks'],
                'velocity': note['velocity'],
                'in_chord': note['in_chord'],
                'chord_id': note['chord_id']
            }

            # Ajouter des attributs dérivés pour la durée
            if note['duration_ticks'] < 100:
                symbol['duration_category'] = 'très court'
            elif note['duration_ticks'] < 300:
                symbol['duration_category'] = 'court'
            elif note['duration_ticks'] < 600:
                symbol['duration_category'] = 'moyen'
            elif note['duration_ticks'] < 1000:
                symbol['duration_category'] = 'long'
            else:
                symbol['duration_category'] = 'très long'

            symbols.append(symbol)

        # Création de la structure de données finale
        result = {
            'file_name': os.path.basename(midi_file),
            'full_path': midi_file,
            'ticks_per_beat': mid.ticks_per_beat,
            'num_notes': len(notes),
            'num_chords': len(chords),
            'symbols': symbols
        }

        return result
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {midi_file}: {str(e)}")
        return None

def find_midi_files(base_dir):
    """
    Recherche récursivement tous les fichiers MIDI dans un répertoire.
    Args:
        base_dir: Répertoire de base à parcourir
    Returns:
        Liste des chemins de fichiers MIDI trouvés
    """
    midi_files = []
    print(f"Recherche de fichiers MIDI dans {base_dir}...")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                midi_path = os.path.join(root, file)
                midi_files.append(midi_path)
    print(f"Total de {len(midi_files)} fichiers MIDI trouvés.")
    return midi_files

def process_dataset(dir, output_dir, limit=None):
    """
    Traite l'ensemble de la base de données et stocke tous les symboles dans un seul fichier JSON.
    Args:
        dir: Répertoire contenant les fichiers MIDI
        output_dir: Répertoire où sauvegarder les features extraites
        limit: Nombre maximal de fichiers à traiter (None pour tous)
    Returns:
        Dictionnaire des statistiques sur les fichiers traités
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Trouver tous les fichiers MIDI dans le répertoire
    midi_files = find_midi_files(dir)

    if limit is not None:
        midi_files = midi_files[:limit]
        print(f"Traitement limité à {limit} fichiers.")

    if not midi_files:
        print(f"Aucun fichier MIDI trouvé dans {dir} et ses sous-répertoires.")
        return {}

    # Statistiques globales
    stats = {
        'total_files': len(midi_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_notes': 0,
        'total_chords': 0,
        'start_time': time.time()
    }

    # Structure pour contenir tous les symboles
    all_data = {
        'files': []
    }

    # Ensemble pour suivre les symboles uniques (par leurs attributs pertinents)
    unique_symbols = set()

    with tqdm(total=len(midi_files), desc="Traitement de l'ensemble des fichiers MIDI") as pbar:
        # Traiter chaque fichier MIDI
        for midi_file in midi_files:
            try:
                # Extraire les caractéristiques
                features = extract_features(midi_file)

                if features:
                    # Ajouter les features à notre collection globale
                    all_data['files'].append(features)

                    # Ajouter un séparateur après les symboles de ce fichier
                    all_data['files'].append({"separator": True})

                    # Mettre à jour les statistiques
                    stats['processed_files'] += 1
                    stats['total_notes'] += features['num_notes']
                    stats['total_chords'] += features['num_chords']

                    # Ajouter les symboles uniques à notre ensemble
                    for symbol in features['symbols']:
                        # Créer une représentation hashable des attributs importants
                        # Seulement pitch, duration, in_chord et velocity
                        symbol_hash = (
                            symbol['pitch'],
                            symbol['duration'],
                            symbol['velocity'],
                            symbol['in_chord'],
                            symbol['duration_category']
                        )
                        unique_symbols.add(symbol_hash)
                else:
                    stats['failed_files'] += 1

            except Exception as e:
                print(f"\nErreur lors du traitement du fichier {midi_file}: {str(e)}")
                stats['failed_files'] += 1

            # Mettre à jour la barre de progression
            pbar.update(1)

    # Finaliser les statistiques
    stats['elapsed_time'] = time.time() - stats['start_time']
    stats['avg_notes_per_file'] = stats['total_notes'] / stats['processed_files'] if stats['processed_files'] > 0 else 0
    stats['avg_chords_per_file'] = stats['total_chords'] / stats['processed_files'] if stats['processed_files'] > 0 else 0
    stats['total_unique_symbols'] = len(unique_symbols)
    stats['unique_symbols_ratio'] = len(unique_symbols) / stats['total_notes'] if stats['total_notes'] > 0 else 0

    # Sauvegarder toutes les données dans un seul fichier JSON
    print("Sauvegarde des données dans un fichier JSON unique...")
    with open(os.path.join(output_dir, "all_symbols.json"), 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Sauvegarder les statistiques dans un fichier JSON séparé
    with open(os.path.join(output_dir, "stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nTraitement terminé en {stats['elapsed_time']:.1f} secondes.")
    print(f"Fichiers traités: {stats['processed_files']}/{stats['total_files']}")
    print(f"Fichiers en échec: {stats['failed_files']}")
    print(f"Total de notes extraites: {stats['total_notes']}")
    print(f"Total d'accords identifiés: {stats['total_chords']}")
    print(f"Nombre de symboles uniques: {stats['total_unique_symbols']}")
    print(f"Ratio de symboles uniques: {stats['unique_symbols_ratio']:.4f} ({stats['total_unique_symbols']}/{stats['total_notes']})")
    print(f"Moyenne de notes par fichier: {stats['avg_notes_per_file']:.1f}")
    print(f"Moyenne d'accords par fichier: {stats['avg_chords_per_file']:.1f}")

    return stats

# Chemin vers le répertoire contenant les fichiers MIDI de Maestro
input_dir = "dataset/maestro-v3.0.0-midi"

# Chemin vers le répertoire où sauvegarder les fichiers JSON
output_dir= "FO"

# Traitement des fichiers MIDI
if __name__ == "__main__":
    stats = process_dataset(input_dir, output_dir)

    
    all_symbols_file = os.path.join(output_dir, "all_symbols.json")
    if os.path.exists(all_symbols_file):
        print(f"\nFichier JSON global créé: {all_symbols_file}")