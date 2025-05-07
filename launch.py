import mido
import time
import fluidsynth

def list_midi_devices():
    """List available MIDI input and output devices."""
    print("Available MIDI devices:")
    print("\nInput ports:")
    for name in mido.get_input_names():
        print("- " + name)
    
    print("\nOutput ports:")
    for name in mido.get_output_names():
        print("- " + name)

def select_device(port_list, port_type="input"):
    """Let user select a MIDI device from the list."""
    if not port_list:
        print(f"No MIDI {port_type} ports available.")
        return None
    
    for i, port in enumerate(port_list):
        print(f"{i+1}: {port}")
    
    while True:
        try:
            selection = int(input(f"Select {port_type} port number: "))
            if 1 <= selection <= len(port_list):
                return port_list[selection - 1]
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

def modify_note(msg):
    """
    Modify a MIDI note message.
    This is a sample function - customize based on your needs.
    """
    if msg.type == 'note_on' or msg.type == 'note_off':
        # Example: Transpose up by one octave (12 semitones)
        new_note = (msg.note + 12) % 128
        
        # Example: Modify velocity (make it louder)
        new_velocity = min(127, int(msg.velocity * 1.2)) if msg.type == 'note_on' else msg.velocity
        
        # Create a new message with modified parameters
        return msg.copy(note=new_note, velocity=new_velocity)
    
    # Handle other types of MIDI messages you might want to modify
    elif msg.type == 'control_change':
        # Example: Echo control changes (like modulation wheel, sustain pedal)
        return msg.copy()
    elif msg.type == 'program_change':
        # Example: Change instrument selection
        # Return the original message or modify program number
        return msg.copy()
    elif msg.type == 'pitchwheel':
        # Example: Amplify pitch bend values
        new_pitch = max(-8192, min(8191, int(msg.pitch * 1.5)))
        return msg.copy(pitch=new_pitch)
    
    # Return original message for other message types
    return msg

# Main program
print("MIDI Signal Processor")
list_midi_devices()

# Select input device
input_ports = mido.get_input_names()
if not input_ports:
    print("No MIDI input devices found.")
    exit()

print("\nSelect your MIDI input device:")
for i, port in enumerate(input_ports):
    print(f"{i+1}: {port}")

selected = 0
while selected < 1 or selected > len(input_ports):
    try:
        selected = int(input("Enter number: "))
    except ValueError:
        print("Please enter a valid number.")

input_port = input_ports[selected-1]
print(f"Selected input: {input_port}")

# Ask about FluidSynth
use_fluidsynth = input("\nDo you want to play sounds using FluidSynth? (y/n): ").lower().startswith('y')

# Soundfont path
soundfont_path = "/home/sylogue/stage/Roland_SC-88.sf2"  # Default path
if use_fluidsynth:
    custom_path = input(f"\nEnter soundfont path (press Enter for default: {soundfont_path}): ")
    if custom_path.strip():
        soundfont_path = custom_path

# Process MIDI
try:
    # Open input port
    midi_input = mido.open_input(input_port)
    print(f"Listening to MIDI input from: {input_port}")
    
    # Set up FluidSynth for sound output
    fs = None
    if use_fluidsynth:
        try:
            fs = fluidsynth.Synth()
            fs.start(driver="pulseaudio")
            # Load the SoundFont
            sfid = fs.sfload(soundfont_path)
            # Set instrument (General MIDI program 50 - Synth Strings 1)
            fs.program_select(0, sfid, 0, 50)
            print(f"FluidSynth initialized with soundfont: {soundfont_path}")
        except Exception as e:
            print(f"Error initializing FluidSynth: {e}")
            fs = None
    
    print("\nPress keys on your MIDI device. Press Ctrl+C to stop...\n")
    
    # Process incoming messages
    for msg in midi_input:
        # Print original message
        print(f"Received: {msg}")
        
        # Modify message
        modified_msg = modify_note(msg)
        
        # Print modified message if it was changed
        if modified_msg != msg:
            print(f"Modified to: {modified_msg}")
        
        # Play through FluidSynth
        if fs and (modified_msg.type == 'note_on' or modified_msg.type == 'note_off'):
            if modified_msg.type == 'note_on' and modified_msg.velocity > 0:
                fs.noteon(0, modified_msg.note, modified_msg.velocity)
            else:
                fs.noteoff(0, modified_msg.note)
        
except KeyboardInterrupt:
    print("\nMIDI processing stopped.")
finally:
    # Clean up
    if 'midi_input' in locals() and midi_input:
        midi_input.close()
    if 'fs' in locals() and fs:
        fs.delete()