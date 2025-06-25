import pathlib
import random
import numpy as np
import json
import torch
import torch.nn.functional as F
import model as md
from midi_processor import MidiSymbolProcessor
import matplotlib.pyplot as plt

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
    "max_num_steps": 50000,
    "early_stopping": 5000,
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

midi_folder = "/home/sylogue/midi_xml/Weimar_jazz_database"
dataset_name = pathlib.Path(midi_folder).name
cache_path = run_dir / f"{dataset_name}_performances.json"

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
             n['pitch'] - md.PIANO_LOWEST_KEY_MIDI_PITCH,
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


# Subsamples performances to create a minibatch
def performances_to_batch(performances, device, train=True):
    batch_k = []
    batch_t = []
    for p in performances:
        # Subsample seq_len notes from performance
        assert len(p) >= CFG["seq_len"]
        if train:
            subsample_offset = random.randrange(0, len(p) - CFG["seq_len"])
        else:
            subsample_offset = 0
        subsample = p[subsample_offset : subsample_offset + CFG["seq_len"]]
        assert len(subsample) == CFG["seq_len"]

        # Data augmentation
        if train:
            stretch_factor = random.random() * CFG["data_augment_time_stretch_max"] * 2
            stretch_factor += 1 - CFG["data_augment_time_stretch_max"]
            transposition_factor = random.randint(
                -CFG["data_augment_transpose_max"], CFG["data_augment_transpose_max"]
            )
            subsample = [
                (
                    n[0] * stretch_factor,
                    n[1] * stretch_factor,
                    max(0, min(n[2] + transposition_factor, md.PIANO_NUM_KEYS - 1)),
                    n[3],
                )
                for n in subsample
            ]
        
        # Key features
        batch_k.append([n[2] for n in subsample])

        # Onset features
        # NOTE: For stability, we pass delta time to Piano Genie instead of time.
        t = np.diff([n[0] for n in subsample])
        t = np.concatenate([[1e8], t])
        t = np.clip(t, 0, CFG["data_delta_time_max"])
        batch_t.append(t)

    return (torch.tensor(batch_k).long(), torch.tensor(batch_t).float())

step = 0
best_eval_loss = float("inf")

#early stopping 
last_improvement = 0

# storing losses for ploting
train_rec_losses = []
train_margin_losses = []
train_contour_losses = []
train_total_losses   = []

eval_rec_losses  = []
eval_contour_rates = []

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
            torch.save(model.state_dict(), run_dir / f"model_{dataset_name}.pt")
            best_eval_loss = avg_rec
            last_improvement = step 

        eval_rec_losses.append(avg_rec)
        eval_contour_rates.append(float(np.mean(all_viol)))
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

    # early‐stopping check
    if step - last_improvement >= CFG["early_stopping"]:
        print(f"Stopping early at step {step}: no eval improvement " +
              f"in the last {CFG["early_stopping"]} steps (since step {last_improvement}).")
        break

    train_rec_losses.append(loss_recons.item())
    train_margin_losses.append(loss_margin.item())
    train_contour_losses.append(loss_contour.item())
    train_total_losses.append(loss.item())

    if step % CFG["summarize_frequency"] == 0:
        print(
            f"{step:6d} TRAIN → rec={loss_recons.item():.4f}, "
            f"margin={loss_margin.item():.4f}, contour={loss_contour.item():.4f}"
        )

print(f"Best model saved to: {run_dir/f"model_{dataset_name}.pt"}")


steps = range(1, len(train_rec_losses) + 1)
plt.figure()
plt.plot(steps, train_rec_losses,     label='Reconstruction')
plt.plot(steps, train_margin_losses,  label='Margin')
plt.plot(steps, train_contour_losses, label='Contour')
plt.plot(steps, train_total_losses,   label='Total')  
plt.xlabel(f'Buckets of {CFG["summarize_frequency"]} steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curves')
train_path = run_dir / f"{dataset_name}_train_losses.png"
plt.savefig(train_path, dpi=300)
plt.close()

# --- Evaluation curves ---
eval_steps = [i * CFG['eval_frequency'] for i in range(len(eval_rec_losses))]
plt.figure()
plt.plot(eval_steps, eval_rec_losses,    marker='o', label='Eval Reconstruction')
plt.plot(eval_steps, eval_contour_rates, marker='x', label='Contour Violation Rate')
plt.xlabel('Training Step')
plt.ylabel('Metric')
plt.legend()
plt.title('Evaluation Metrics Over Time')
eval_path = run_dir / f"{dataset_name}_eval_metrics.png"
plt.savefig(eval_path, dpi=300)
plt.close()

print(f"Saved training curves to {train_path}")
print(f"Saved evaluation curves to {eval_path}")