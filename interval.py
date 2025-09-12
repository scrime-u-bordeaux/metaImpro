from collections import defaultdict

def midi_to_pitch_class(chord_map):
    pitchclass_map = {}
    for chord, pitches in chord_map.items():
        pitchclass = [pitch % 12 for pitch in pitches]
        pitchclass_map[chord] = pitchclass
    return pitchclass_map
    
def build_interval_dict(chord_map):
    interval_dict = {}
    for chord, pitches in chord_map.items():
        pitches = list(pitches)
        intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
        interval_dict[chord] = intervals
    return interval_dict

def build_interval_table(data, max_order):
    table = defaultdict(lambda: defaultdict(int))
    for seq in data.values():
        seq = list(seq)
        for i in range(len(seq)):
            succ = seq[i]
            for order in range(1, max_order + 1):
                if i - order < 0:
                    break
                ctx = tuple(seq[i-order:i])
                table[ctx][succ] += 1
    return {ctx: dict(d) for ctx, d in table.items()}

