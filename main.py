import os
import random
from factor_oracle import FactorOracle, process_midi_data_with_oracle
from improvization import FOGenerator, process_maestro_file, play_generated_sequence
# from create_symbols import process_dataset  #  Décommenter seulement si vous voulez extraire les symboles à nouveau !

def main():
    """Main function: charge l'Oracle, génère l'improvisation depuis un fichier MIDI de Maestro."""

    # ⚠️ Étape 1 : Extraction des symboles MIDI (à faire uniquement la première fois)
    # midi_dataset_dir = "dataset/maestro-v3.0.0-midi"
    # output_symbol_dir = "FO"
    
    # if not os.path.exists(output_symbol_dir):
    #     os.makedirs(output_symbol_dir)

    # print("Extraction des symboles des fichiers MIDI...")
    # stats = process_dataset(midi_dataset_dir, output_symbol_dir)
    # print(f"Extraction terminée. {stats['processed_files']} fichiers MIDI traités.")

    # Étape 2 : Vérifier et charger l'Oracle des Facteurs
    oracle_file = "FO/oracle_results/factor_oracle_1.json"
    
    if not os.path.exists(oracle_file):
        print(f"Oracle non trouvé. Construction en cours...")
        
        json_symbol_file = "FO/all_symbols.json"
        oracle_output_dir = "FO/oracle_results"

        if not os.path.exists(json_symbol_file):
            print(f"Erreur: Fichier {json_symbol_file} non trouvé. Vérifiez que l'extraction des symboles a été faite.")
            return

        print("Construction de l'Oracle des Facteurs...")
        oracle, symbol_count = process_midi_data_with_oracle(json_symbol_file, oracle_output_dir, limit=1)
        print(f"Oracle des Facteurs créé avec succès! {symbol_count} symboles appris.")

    # Charger l'Oracle maintenant qu'il est construit
    oracle = FactorOracle.load(oracle_file)
    print(f"Oracle chargé avec {len(oracle.states)} états.")

    # Étape 3 : Génération de l'improvisation
    generator = FOGenerator(oracle, continuity_factor=16, taboo_length=8)

    maestro_dir = "dataset/maestro-v3.0.0-midi"
    if os.path.exists(maestro_dir):
        midi_files = []
        for root, _, files in os.walk(maestro_dir):
            for file in files:
                if file.endswith('.midi') or file.endswith('.mid'):
                    midi_files.append(os.path.join(root, file))
        if midi_files:
            test_file = random.choice(midi_files)
            print(f"Utilisation du fichier {test_file} comme source d'inputs aplatis...")
            input_symbols = process_maestro_file(test_file, oracle)
            if not input_symbols:
                print("Aucun symbole extrait du fichier.")
                return
            
            sequence = generator.generate_sequence(seed_symbols=input_symbols, length=100)
            output_file = "maestro_based_generation.mid"
            play_generated_sequence(sequence, output_file, inter_onset=100)
        else:
            print("Aucun fichier MIDI trouvé dans le dossier Maestro.")
    else:
        print(f"Dossier Maestro {maestro_dir} non trouvé.")
    
    print("Génération terminée!")

if __name__ == "__main__":
    main()
