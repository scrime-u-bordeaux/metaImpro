import mido

# Affiche la liste des ports d'entrée MIDI disponibles
print("Input ports:")
for name in mido.get_input_names():
    print("  ", name)

# Affiche la liste des ports de sortie MIDI disponibles
print("Output ports:")
for name in mido.get_output_names():
    print("  ", name)
"""
import pygame.midi

pygame.midi.init()
count = pygame.midi.get_count()
print("Nombre de devices MIDI détectés :", count)

for i in range(count):
    interf, name, is_input, is_output, opened = pygame.midi.get_device_info(i)
    io = []
    if is_input:
        io.append("IN")
    if is_output:
        io.append("OUT")
    print(f"  id={i}  name={name.decode()}  ({','.join(io)})")

pygame.midi.quit()"""