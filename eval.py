# analysis.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# --- config ---
DATA_FILE = "eval/accomp_notes_20250922_092434.json"
OUT_DIR = Path("analysis_output")
OUT_DIR.mkdir(exist_ok=True)

# --- helper ---
def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # normalize to DataFrame
    rows = []
    for e in data:
        r = {}
        r['chord'] = e.get('chord')
        # pitch can be list: use first pitch for analysis
        pitch = e.get('pitch')
        r['pitch'] = pitch[0] if isinstance(pitch, list) and len(pitch)>0 else (pitch or np.nan)
        r['onset'] = e.get('onset', np.nan)
        r['duration'] = e.get('duration', np.nan)
        r['velocity'] = e.get('velocity', np.nan)
        r['effective_duration'] = e.get('effective_duration', np.nan)
        r['is_black'] = bool(e.get('is_black')) if 'is_black' in e else np.nan
        r['desired'] = e.get('desired', np.nan)
        r['actual'] = e.get('actual', np.nan)
        r['success'] = e.get('success', None)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df

df = load_json(DATA_FILE)

# basic cleaning / types
df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
df['pitch'] = pd.to_numeric(df['pitch'], errors='coerce')
df['effective_duration'] = pd.to_numeric(df['effective_duration'], errors='coerce')
df['velocity'] = pd.to_numeric(df['velocity'], errors='coerce')
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

# global metrics
n_total = len(df)
t_start = df['onset'].min()
t_end = (df['onset'] + df['duration']).max()
total_duration = t_end - t_start
density = n_total / total_duration if total_duration>0 else np.nan

print(f"Total notes: {n_total}")
print(f"Session duration: {total_duration:.2f} s (onset from {t_start} to {t_end})")
print(f"Notes per second: {density:.3f}")

# per-chord summary
agg_funcs = {
    'pitch': ['count', 'mean'],
    'effective_duration': ['mean','std'],
    'velocity': ['mean','std'],
    'is_black': ['sum','count'],
    'success': [lambda x: x.eq(True).sum(), 'count']
}
per_chord = df.groupby('chord').agg(
    n_notes = ('pitch','count'),
    mean_pitch = ('pitch','mean'),
    mean_effective_duration = ('effective_duration','mean'),
    std_effective_duration = ('effective_duration','std'),
    mean_velocity = ('velocity','mean'),
    std_velocity = ('velocity','std'),
    n_black = ('is_black', lambda x: int(x.eq(True).sum())),
    pct_black = ('is_black', lambda x: x.eq(True).sum() / max(1, x.count())),
    n_success = ('success', lambda x: int(x.eq(True).sum())),
    pct_success = ('success', lambda x: int(x.eq(True).sum()) / max(1, x.count()))
).reset_index()

per_chord.to_csv(OUT_DIR / "summary_per_chord.csv", index=False)
print("\nPer-chord summary saved to:", OUT_DIR / "summary_per_chord.csv")
print(per_chord)

# global black vs white success
black_df = df[df['is_black']==True]
white_df = df[df['is_black']==False]

def success_rate(subdf):
    if subdf.empty: return np.nan
    return subdf['success'].eq(True).sum() / max(1, subdf['success'].count())

sr_black = success_rate(black_df)
sr_white = success_rate(white_df)
print(f"\nSuccess rate (black keys): {sr_black:.3f}")
print(f"Success rate (white keys): {sr_white:.3f}")

# intervals distribution (pitch differences)
df_sorted = df.sort_values('onset').reset_index(drop=True)
df_sorted['next_pitch'] = df_sorted['pitch'].shift(-1)
df_sorted['interval'] = df_sorted['next_pitch'] - df_sorted['pitch']
interval_counts = df_sorted['interval'].dropna().value_counts().sort_index()

# save interval counts
interval_counts.to_csv(OUT_DIR / "interval_counts.csv", header=['count'])
print("\nInterval counts saved to:", OUT_DIR / "interval_counts.csv")

# success vs failure stats for effective_duration & velocity
stats = []
for label, sub in [('success', df[df['success']==True]), ('failure', df[df['success']==False])]:
    stats.append({
        'label': label,
        'n': len(sub),
        'mean_eff_dur': sub['effective_duration'].mean(),
        'std_eff_dur': sub['effective_duration'].std(),
        'mean_vel': sub['velocity'].mean(),
        'std_vel': sub['velocity'].std()
    })
stats_df = pd.DataFrame(stats)
stats_df.to_csv(OUT_DIR / "success_vs_failure_stats.csv", index=False)
print("\nSuccess vs failure stats saved to:", OUT_DIR / "success_vs_failure_stats.csv")
print(stats_df)

# --- Figures ---

# 1) histogram of pitches
plt.figure()
plt.hist(df['pitch'].dropna(), bins=range(int(df['pitch'].min())-1, int(df['pitch'].max())+2))
plt.title("Histogram of pitches")
plt.xlabel("MIDI pitch")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_pitches.png")
plt.close()

# 2) timeline raster: onset vs pitch, marker for success
plt.figure(figsize=(10,4))
suc = df[df['success']==True]
fail = df[df['success']==False]
plt.scatter(suc['onset'], suc['pitch'], marker='o', label='success')
plt.scatter(fail['onset'], fail['pitch'], marker='x', label='failure')
plt.xlabel("Onset (s)")
plt.ylabel("Pitch (MIDI)")
plt.legend()
plt.title("Timeline: pitch vs onset (success vs failure)")
plt.tight_layout()
plt.savefig(OUT_DIR / "timeline_success_fail.png")
plt.close()

# 3) boxplots effective_duration by success/failure
plt.figure()
data_to_plot = [df[df['success']==True]['effective_duration'].dropna(), df[df['success']==False]['effective_duration'].dropna()]
plt.boxplot(data_to_plot, labels=['success','failure'])
plt.ylabel("effective_duration (s)")
plt.title("Effective duration: success vs failure")
plt.tight_layout()
plt.savefig(OUT_DIR / "box_effdur_success_fail.png")
plt.close()

# 4) bar: pct success per chord
plt.figure(figsize=(8,4))
p = per_chord.sort_values('pct_success', ascending=False)
plt.bar(p['chord'].astype(str), p['pct_success'])
plt.xlabel("Chord")
plt.ylabel("Pct success")
plt.title("Pct succès par accord")
plt.tight_layout()
plt.savefig(OUT_DIR / "pct_success_per_chord.png")
plt.close()

print("\nFigures saved in:", OUT_DIR)

# Save cleaned dataframe for manual inspection
df.to_csv(OUT_DIR / "cleaned_data.csv", index=False)
print("Cleaned data saved to:", OUT_DIR / "cleaned_data.csv")

# Quick summary file for report
report_summary = {
    'total_notes': n_total,
    'session_duration_s': float(total_duration),
    'notes_per_second': float(density),
    'global_success_rate': float(df['success'].eq(True).sum() / max(1, df['success'].count()))
}
pd.Series(report_summary).to_csv(OUT_DIR / "report_summary.csv")
print("Report summary saved to:", OUT_DIR / "report_summary.csv")
