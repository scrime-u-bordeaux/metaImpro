import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- config ---
DATA_FILE = Path("eval/")
gesture_path = "eval/fixed_gesture"
OUT_DIR = Path("eval/analysis_output")
OUT_DIR.mkdir(exist_ok=True)

def json_to_df(path):
    json_files = [f for f in os.listdir(path) if f.endswith('.json')]
    dfs = []
    # Lire chaque fichier JSON et le stocker dans la liste
    for file in json_files:
        file_path = os.path.join(path, file)
        df = pd.read_json(file_path)
        df.name = file_path
        dfs.append(df)
    return dfs

def get_mean_and_median(dfs):
    results = []
    for idx, df in enumerate(dfs):
        df['pitch_values'] = df['pitch'].apply(lambda x: x[0])
        mean_pitch = df['pitch_values'].mean()
        median_pitch = df['pitch_values'].median()
        
        # Calculer le taux de succès du contour mélodique
        success_rate = df['success'].mean() * 100 if 'success' in df.columns else None
        
        results.append((os.path.basename(df.name), mean_pitch, median_pitch, success_rate))
        print(f"Fichier {df.name}: Moyenne des pitches = {mean_pitch:.2f}, Médiane des pitches = {median_pitch:.2f}")
        if success_rate is not None:
            print(f"  -> Taux de succès contour: {success_rate:.1f}%")
    
    # Créer un DataFrame pour affichage propre
    columns = ['Fichier', 'Moyenne', 'Médiane']
    if results and results[0][3] is not None:
        columns.append('Taux_Succès_%')
        
    stats_df = pd.DataFrame(results, columns=columns)
    stats_df['Moyenne'] = stats_df['Moyenne'].round(2)
    stats_df['Médiane'] = stats_df['Médiane'].round(2)
    if 'Taux_Succès_%' in stats_df.columns:
        stats_df['Taux_Succès_%'] = stats_df['Taux_Succès_%'].round(1)
    
    return results, stats_df

def get_distances(dfs):
    distances = []
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            # Extraire les valeurs des pitches pour chaque DataFrame
            a = dfs[i]
            b = dfs[j]
            # S'assurer que les DataFrames ont la même longueur
            if len(a) != len(b):
                print(f"Les DataFrames {a.name} et {b.name} n'ont pas la même longueur, impossible de calculer la distance.")
                continue
            a['pitch_values'] = a['pitch'].apply(lambda x: x[0])
            b['pitch_values'] = b['pitch'].apply(lambda x: x[0])
            # Calculer la distance (différence absolue moyenne)
            distance = (a['pitch_values'] - b['pitch_values']).abs().mean()
            distances.append((os.path.basename(a.name), os.path.basename(b.name), distance))
            print(f"Distance entre {a.name} et {b.name}: {distance:.2f}")
    
    # Créer un DataFrame pour affichage propre
    dist_df = pd.DataFrame(distances, columns=['Fichier 1', 'Fichier 2', 'Distance'])
    dist_df['Distance'] = dist_df['Distance'].round(2)
    
    return distances, dist_df

def create_piano_roll_visualization(dfs):
    """Crée une visualisation piano roll pour tous les fichiers JSON"""
    plt.figure(figsize=(15, 10))
    
    # Définir des couleurs pour chaque fichier
    colors = plt.cm.Set3(np.linspace(0, 1, len(dfs)))
    
    for idx, df in enumerate(dfs):
        filename = os.path.basename(df.name)
        color = colors[idx]
        
        # Extraire les données nécessaires
        df['pitch_values'] = df['pitch'].apply(lambda x: x[0])
        onsets = df['onset']
        pitches = df['pitch_values']
        durations = df['effective_duration'] if 'effective_duration' in df.columns else [0.5] * len(df)
        
        # Dessiner les notes comme des rectangles
        for j, (onset, pitch, duration) in enumerate(zip(onsets, pitches, durations)):
            plt.barh(pitch, duration, left=onset, height=0.8, 
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                    label=filename if j == 0 else "")
    
    plt.xlabel('Temps (s)', fontsize=12)
    plt.ylabel('Hauteur MIDI', fontsize=12)
    plt.title('Piano Roll - Comparaison des fichiers', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig(OUT_DIR / "piano_roll_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt.gcf()

def create_success_rate_chart(stats_df):
    """Crée un graphique du taux de succès si disponible"""
    if 'Taux_Succès_%' not in stats_df.columns:
        print("Pas de données de taux de succès disponibles")
        return None
        
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(stats_df)), stats_df['Taux_Succès_%'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(stats_df))))
    
    plt.xlabel('Fichiers', fontsize=12)
    plt.ylabel('Taux de succès (%)', fontsize=12)
    plt.title('Taux de succès du contour mélodique par fichier', fontsize=14, fontweight='bold')
    plt.xticks(range(len(stats_df)), stats_df['Fichier'], rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig(OUT_DIR / "taux_succes.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return plt.gcf()

dfs = json_to_df(gesture_path)
mean_med, stats_table = get_mean_and_median(dfs)
dist, dist_table = get_distances(dfs)
create_piano_roll_visualization(dfs)
create_success_rate_chart(dfs)

# Affichage des tableaux propres
print("\n" + "="*50)
print("TABLEAU DES STATISTIQUES")
print("="*50)
print(stats_table.to_string(index=False))

print("\n" + "="*50)
print("TABLEAU DES DISTANCES")
print("="*50)
print(dist_table.to_string(index=False))

# Sauvegarde des tableaux
stats_table.to_csv(OUT_DIR / "statistiques_pitch.csv", index=False)
dist_table.to_csv(OUT_DIR / "distances_pitch.csv", index=False)

# Pour LaTeX (si vous utilisez LaTeX dans votre rapport)
with open(OUT_DIR / "stats_latex.txt", "w") as f:
    f.write(stats_table.to_latex(index=False))

with open(OUT_DIR / "distances_latex.txt", "w") as f:
    f.write(dist_table.to_latex(index=False))

print(f"\nTableaux sauvegardés dans {OUT_DIR}/")
print("- statistiques_pitch.csv")
print("- distances_pitch.csv") 
print("- stats_latex.txt (pour LaTeX)")
print("- distances_latex.txt (pour LaTeX)")