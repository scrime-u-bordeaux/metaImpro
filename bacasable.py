import fluidsynth
import time

# Test simple de FluidSynth
def test_fluidsynth():
    print("Initialisation de FluidSynth...")
    fs = fluidsynth.Synth()
    print("Démarrage du synthétiseur...")
    fs.start(driver="alsa")
    
    print("Chargement de la SoundFont...")
    sfid = fs.sfload("/home/sylogue/stage/Roland_SC-88.sf2")
    print(f"SoundFont chargée avec ID: {sfid}")
    
    print("Sélection du programme...")
    fs.program_select(0, sfid, 0, 0)
    
    print("Joue Do médian (note 60)...")
    fs.noteon(0, 60, 100)
    time.sleep(1)
    fs.noteoff(0, 60)
    
    print("Joue Mi médian (note 64)...")
    fs.noteon(0, 64, 100)
    time.sleep(1)
    fs.noteoff(0, 64)
    
    print("Joue Sol médian (note 67)...")
    fs.noteon(0, 67, 100)
    time.sleep(1)
    fs.noteoff(0, 67)
    
    print("Nettoyage...")
    fs.delete()
    print("Test terminé.")

if __name__ == "__main__":
    test_fluidsynth()