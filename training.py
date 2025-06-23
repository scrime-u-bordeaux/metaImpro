import pathlib
import random
import numpy as np
import json
import torch
import torch.nn.functional as F
import model as md
from midi_processor import MidiSymbolProcessor

CFG = {
    "seed": 0,
    "num_buttons": 8,
    "data_delta_time_max": 1.0,
    "data_augment_time_stretch_max": 0.05,
    "data_augment_transpose_max": 6,
    "model_rnn_dim": 128,
    "model_rnn_num_layers": 2,
    "batch_size": 32,
    "seq_len": 128,
    "lr": 3e-4,
    "loss_margin_multiplier": 1.0,
    "loss_contour_multiplier": 1.0,
    "summarize_frequency": 128,
    "eval_frequency": 128,
    "max_num_steps": 50000
}

run_dir = pathlib.Path("piano_genie")
run_dir.mkdir(exist_ok=True)
with open(run_dir / "cfg.json", "w") as f:
    json.dump(CFG, f, indent=2)

processor = MidiSymbolProcessor(
    inner_chord_threshold_ms=50.0,
    maximum_chord_threshold_ms=200.0,
    include_chords=False
)

midi_folder = "/home/sylogue/stage/dataset/maestro-v3.0.0-midi"
cache_path = run_dir / f"{pathlib.Path(midi_folder).name}_performances.json"

# --- Build or load cached performances ---
if cache_path.exists():
    print(f"Loading cached performances from {cache_path}...")
    with open(cache_path, "r") as f:
        all_performances = json.load(f)
else:
    print("Building performances from MIDI files...")
    all_performances = []
    for midi_file in processor.find_midi_files(midi_folder):
        notes = processor.extract_notes(midi_file)
        perf = [
            (n['onset'] / 1000.0,
             n['duration'] / 1000.0,
             n['pitch'],
             n['velocity'])
            for n in notes
        ]
        if len(perf) >= CFG['seq_len']:
            all_performances.append(perf)
    print(f"Extracted {len(all_performances)} performances; caching to disk...")
    with open(cache_path, "w") as f:
        json.dump(all_performances, f)

# Shuffle + split
random.seed(CFG['seed'])
random.shuffle(all_performances)
num_val = int(0.1 * len(all_performances))
DATASET = {
    "train": all_performances[num_val:],
    "validation": all_performances[:num_val]
}

# Set seeds
if CFG["seed"] is not None:
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = md.PianoGenieAutoencoder(CFG).to(device).train()

print("-" * 80)
for n, p in model.named_parameters():
    print(f"{n}, {tuple(p.shape)}")

optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

def performances_to_batch(performances, device, train=True):
    """Turns a list of performances into (keys, delta_times) tensors."""
    batch_k = []
    batch_t = []
    for p in performances:
        # pick a subsequence
        if train:
            offset = random.randrange(0, len(p) - CFG["seq_len"])
        else:
            offset = 0
        subseq = p[offset : offset + CFG["seq_len"]]

        # optionally augment and always clamp pitches
        if train:
            sf = 1 + (random.random() * 2 - 1) * CFG["data_augment_time_stretch_max"]
            tr = random.randint(
                -CFG["data_augment_transpose_max"],
                CFG["data_augment_transpose_max"]
            )
        keys = []
        for onset, dur, pitch, vel in subseq:
            if train:
                pitch = pitch + tr
            # clamp into [0, PIANO_NUM_KEYS-1]
            pitch = max(0, min(pitch, md.PIANO_NUM_KEYS - 1))
            keys.append(pitch)
        batch_k.append(keys)

        # compute clipped delta-times
        times = np.array([onset for onset, *_ in subseq], dtype=np.float32)
        dt = np.diff(times, prepend=times[0])
        dt = np.clip(dt, 0.0, CFG["data_delta_time_max"])
        batch_t.append(dt)

    # stack to a single numpy array (avoids the slow warning) then to tensor
    k_np = np.array(batch_k, dtype=np.int64)
    t_np = np.stack(batch_t).astype(np.float32)

    return (
        torch.from_numpy(k_np).to(device),
        torch.from_numpy(t_np).to(device),
    )

step = 0
best_eval_loss = float("inf")

while CFG["max_num_steps"] is None or step < CFG["max_num_steps"]:
    # --- evaluation ---
    if step % CFG["eval_frequency"] == 0:
        model.eval()
        all_rec, all_viol = [], []
        with torch.no_grad():
            for i in range(0, len(DATASET["validation"]), CFG["batch_size"]):
                vk, vt = performances_to_batch(
                    DATASET["validation"][i : i + CFG["batch_size"]],
                    device, train=False
                )
                vhat_k, v_e = model(vk, vt)
                v_b = model.quant.real_to_discrete(v_e)

                # reconstruction loss
                rec = F.cross_entropy(
                    vhat_k.view(-1, md.PIANO_NUM_KEYS),
                    vk.view(-1),
                    reduction="none"
                ).cpu().numpy()
                all_rec.extend(rec)

                # contour violations
                viol = (
                    torch.sign(torch.diff(vk, 1)) !=
                    torch.sign(torch.diff(v_b, 1))
                ).float().cpu().numpy()
                all_viol.extend(viol)

        avg_rec = float(np.mean(all_rec))
        if avg_rec < best_eval_loss:
            torch.save(model.state_dict(), run_dir / "model.pt")
            best_eval_loss = avg_rec

        print(f"{step:6d} EVAL → rec_loss={avg_rec:.4f}, contour_violation={np.mean(all_viol):.4f}")
        model.train()

    # --- training step ---
    k, t = performances_to_batch(
        random.sample(DATASET["train"], CFG["batch_size"]),
        device, train=True
    )
    optimizer.zero_grad()
    k_hat, e = model(k, t)

    loss_recons = F.cross_entropy(
        k_hat.view(-1, md.PIANO_NUM_KEYS),
        k.view(-1)
    )
    loss_margin = torch.clamp(torch.abs(e) - 1, min=0.0).pow(2).mean()
    loss_contour = torch.clamp(
        1 - torch.diff(k, 1) * torch.diff(e, 1), min=0.0
    ).pow(2).mean()

    loss = (
        loss_recons
        + CFG["loss_margin_multiplier"] * loss_margin
        + CFG["loss_contour_multiplier"] * loss_contour
    )
    loss.backward()
    optimizer.step()
    step += 1

    if step % CFG["summarize_frequency"] == 0:
        print(
            f"{step:6d} TRAIN → rec={loss_recons.item():.4f}, "
            f"margin={loss_margin.item():.4f}, contour={loss_contour.item():.4f}"
        )

print(f"Best model saved to: {run_dir/'model.pt'}")
