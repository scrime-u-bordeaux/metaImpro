import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import data_processor as dp
import os
import pickle
from tqdm import tqdm
"""
Classe qui modélise un MLP simple. On donne en entrée le strict minimum, soit la noten la durée et son accord courant
"""


# 1. Définition du vocabulaire factorisé
roots = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
types = ['', '+7', '+79', '+79#', '+7911#', '+79b', '-6', '-69', '-7', '-79', '-7911',
         '-7913', '-79b', '-j7911#', '-j7913', '6', '69', '6911#', '7', '79',
         '79#', '79#11#', '79#13', '7911', '7911#', '7913', '7913b', '79b',
         '79b13', '79b13b', '7alt', 'aug', 'dim', 'dim7', 'j79', 'j79#',
         'j7911#', 'm', 'm7b5', 'maj7', 'min', 'minmaj7', 'sus', 'sus7',
         'sus79', 'sus7913']

root2idx = {r: i for i, r in enumerate(roots)}
type2idx = {t: i for i, t in enumerate(types)}

# 2. Dataset Skip-gram factorisé
class Chord2VecIterableDataset(IterableDataset):
    """
    Iterable dataset qui génère les paires (target, context, label)
    à la volée sans stocker tout en mémoire.
    """
    def __init__(self, sequences, window_size=2, neg_samples=5):
        self.sequences = sequences
        self.window = window_size
        self.neg = neg_samples
        self.V_r = len(roots)
        self.V_t = len(types)

    def __iter__(self):
        # Itérer chaque séquence
        for seq in self.sequences:
            idx_seq = [(root2idx.get(r,0), type2idx.get(t,0)) for r, t in seq]
            L = len(idx_seq)
            for i, (r_i, t_i) in enumerate(idx_seq):
                # context positions
                left = list(range(max(0, i - self.window), i))
                right = list(range(i+1, min(L, i+1+self.window)))
                for j in left + right:
                    r_j, t_j = idx_seq[j]
                    # paire positive
                    yield r_i, t_i, r_j, t_j, 1
                    # négatifs
                    for _ in range(self.neg):
                        nr = random.randrange(self.V_r)
                        nt = random.randrange(self.V_t)
                        yield r_i, t_i, nr, nt, 0
    

class FactorizedChord2Vec(nn.Module):
    def __init__(self, V_root, V_type, d_r=32, d_t=128):
        super().__init__()
        self.u_root = nn.Embedding(V_root, d_r)
        self.v_root = nn.Embedding(V_root, d_r)
        self.u_type = nn.Embedding(V_type, d_t)
        self.v_type = nn.Embedding(V_type, d_t)
        # Projection optionnelle pour aligner dimensions
        self.proj = nn.Linear(d_r + d_t, d_r + d_t, bias=False)

    def forward(self, rt_i, tp_i, rt_j, tp_j):
        e_ui = torch.cat([self.u_root(rt_i), self.u_type(tp_i)], dim=1)
        e_vj = torch.cat([self.v_root(rt_j), self.v_type(tp_j)], dim=1)
        score = torch.sum(e_ui * e_vj, dim=1)
        return torch.sigmoid(score)


def train_chord2vec_from_raw(raw_data, seq_length=64, epochs=20,
                              batch_size=128, lr=1e-3,
                              window_size=2, neg_samples=3, val_split=0.2):
    # Construire les séquences de (root,type)
    sequences = dp.build_chord_sequences(raw_data, seq_length)
    del raw_data
    # Split train/val
    random.shuffle(sequences)
    split = int(len(sequences) * (1 - val_split))
    train_seqs = sequences[:split]
    val_seqs   = sequences[split:]
    train_ds = Chord2VecIterableDataset(train_seqs, window_size, neg_samples)
    val_ds   = Chord2VecIterableDataset(val_seqs,   window_size, neg_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    print("==============Dataloader_built==============")
    model = FactorizedChord2Vec(len(roots), len(types))
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    history = {'train': [], 'val': []}

    for epoch in tqdm(range(1, epochs+1), "Epochs :"):
        # train
        model.train()
        total_tr = count_tr = 0
        for r_i, t_i, r_j, t_j, lbl in train_loader:
            pred = model(r_i, t_i, r_j, t_j)
            loss = loss_fn(pred, lbl.float())
            opt.zero_grad(); loss.backward(); opt.step()
            total_tr += loss.item(); count_tr += 1
        history['train'].append(total_tr/count_tr)

        # val
        model.eval()
        total_v = count_v = 0
        with torch.no_grad():
            for r_i, t_i, r_j, t_j, lbl in val_loader:
                pred = model(r_i, t_i, r_j, t_j)
                loss = loss_fn(pred, lbl.float())
                total_v += loss.item(); count_v += 1
        history['val'].append(total_v/count_v)

        print(f"Epoch {epoch}: train_loss={history['train'][-1]:.4f}, val_loss={history['val'][-1]:.4f}")

    # Plot
    plt.plot(history['train'], label='train')
    plt.plot(history['val'],   label='val')
    plt.xlabel('Epoch'); plt.ylabel('BCELoss')
    plt.legend()
    plt.savefig(f'chord_model_loss_train_val_with_neg_samples={neg_samples}_seq_length{seq_length}_lr{lr}.png')
    plt.show()

    return model, history


if __name__ == "__main__":

    xml_folder ="/home/sylogue/midi_xml/omnibook_xml"
    db = 'wjazzd.db'
    cache_path = "dataset_final.pkl"

    seq_length=64
    epochs=20
    batch_size=128
    lr=1e-3,
    window_size=2
    neg_samples=3,
    
    if os.path.exists(cache_path):
        print(f"Loading cached dataset_final from {cache_path}")
        with open(cache_path, "rb") as f:
            dataset_final = pickle.load(f)
    else:
        print("Building dataset_final and caching to disk...")
        print("Extracting Weimar data...")
        weimar = dp.extract_all_flat(db)
        print("Processing XML files...")
        omnibook = dp.music_xml_to_data(xml_folder)
        dataset = weimar + omnibook        
        print("Augmenting omnibook...")
        aug1 = dp.augment_dataset(omnibook)
        print("Augmenting weimar...")
        aug2 = dp.augment_dataset(weimar)
        print("Augmenting full dataset...")
        dataset_transposed = dp.augment_dataset(dataset)
        print("Uniformizing dataset...")
        dataset_final = dp.uniformize_dataset(dataset_transposed)
        with open(cache_path, "wb") as f:
            pickle.dump(dataset_final, f)
        print(f"dataset_final saved to {cache_path}")

    model, history = train_chord2vec_from_raw(dataset_final,seq_length=64, epochs=20,
                              batch_size=128, lr=1e-3,
                              window_size=2, neg_samples=3,)
    print("Training completed.")
    # Sauvegarde du modèle
    save_path = f"chord2vec_model_lr={lr}_bs{batch_size}_wp_{window_size}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")