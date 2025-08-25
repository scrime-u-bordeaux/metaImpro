import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from chord_model import FactorizedChord2Vec, type2idx
import pickle

# 1) Chargement et préparation des données
class MelodyDataset(Dataset):
    def __init__(self, performances, seq_len=128, jitter=0.05, dt_max=1.0, train=True):
        """
        performances: liste de performances, chaque perf est une liste de tuples
          (onset, duration, pitch, root_idx, type_idx)
        seq_len: longueur fixe des séquences
        jitter: amplitude max du bruit uniforme ajouté aux delta-times
        dt_max: clip des delta-times entre 0 et dt_max
        """
        self.perf = performances
        self.seq_len = seq_len
        self.jitter = jitter
        self.dt_max = dt_max
        self.train = train

    def __len__(self):
        return len(self.perf)

    def __getitem__(self, idx):
        p = self.perf[idx]
        L = len(p)
        # choisir un segment
        if self.train and L > self.seq_len:
            start = random.randint(0, L - self.seq_len)
        else:
            start = 0
        segment = p[start: start + self.seq_len]
        assert len(segment) == self.seq_len, "Performance trop courte"

        # extraire pitch, onset, duration, chord
        onsets = [n[0] for n in segment]
        pitches= [n[2] for n in segment]
        roots  = [n[3] for n in segment]
        types  = [n[4] for n in segment]

        # calcul des delta times
        dt = []
        prev = onsets[0]
        for o in onsets:
            delta = o - prev
            prev = o
            if self.train:
                delta += random.uniform(-self.jitter, self.jitter)
            dt.append(delta)
        dt = [max(0, min(d, self.dt_max)) for d in dt]

        return {
            'pitch': torch.tensor(pitches, dtype=torch.long),
            'dt':    torch.tensor(dt,     dtype=torch.float),
            'root':  torch.tensor(roots,   dtype=torch.long),
            'type':  torch.tensor(types,   dtype=torch.long),
        }

# 2) Embedding de l'accord via Chord2Vec déjà entraîné
class ChordEmbedding(nn.Module):
    def __init__(self, chord2vec_model):
        super().__init__()
        self.u_root = chord2vec_model.u_root
        self.u_type = chord2vec_model.u_type

    def forward(self, root_idx, type_idx):
        e_r = self.u_root(root_idx)
        e_t = self.u_type(type_idx)
        return torch.cat([e_r, e_t], dim=-1)

# 3) Modèle LSTM conditionnel
class ConditionalLSTM(nn.Module):
    def __init__(self,
                 pitch_vocab_size=88,
                 chord_emb_dim=80,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.input_dim = pitch_vocab_size + 1 + chord_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.out_pitch = nn.Linear(hidden_dim, pitch_vocab_size)
        self.out_dt = nn.Linear(hidden_dim, 1)
        self.contour_proj = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, pitch, dt, chord_emb, hidden=None):
        one_hot = F.one_hot(pitch, num_classes=self.out_pitch.out_features).float()
        x = torch.cat([one_hot, dt.unsqueeze(-1), chord_emb], dim=-1)
        x = self.input_proj(x)
        x, h = self.lstm(x, hidden)
        logits = self.out_pitch(x)
        dt_pred = self.out_dt(x).squeeze(-1)
        e_contour = self.contour_proj(x).squeeze(-1)
        return logits, dt_pred, e_contour, h

# 4) Exemple d'usage
if __name__ == '__main__':
    # chemins vers les fichiers pré-trainés
    CHORD2VEC_PATH = 'chord2vec_model.pth'
    DATASET_PATH    = 'dataset_transposed.pkl'

    # charger dataset pré-transposé
    with open(DATASET_PATH, 'rb') as f:
        dataset_raw = pickle.load(f)

    # DataLoader
    seq_len = 128
    ds_train = MelodyDataset(dataset_raw['train'], seq_len=seq_len, jitter=0.05, train=True)
    ds_val   = MelodyDataset(dataset_raw['val'],   seq_len=seq_len, jitter=0.0,  train=False)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=32, shuffle=False, drop_last=False)

    # charger modèle chord2vec entraîné
    
    chord2vec = FactorizedChord2Vec(V_root=12, V_type=len(type2idx))
    chord2vec.load_state_dict(torch.load(CHORD2VEC_PATH))
    chord2vec.eval()
    chord_emb_model = ChordEmbedding(chord2vec)

    # définir ConditionalLSTM
    chord_dim = chord_emb_model.u_root.embedding_dim + chord_emb_model.u_type.embedding_dim
    model = ConditionalLSTM(pitch_vocab_size=88, chord_emb_dim=chord_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # boucle d'entraînement + validation
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for batch in dl_train:
            pitch = batch['pitch']
            dt    = batch['dt']
            chord_e = chord_emb_model(batch['root'], batch['type'])

            logits, dt_pred, _ = model(pitch, dt, chord_e)
            loss_pitch = F.cross_entropy(logits.view(-1, logits.size(-1)), pitch.view(-1))
            loss_dt    = F.mse_loss(dt_pred, dt)
            loss = loss_pitch + 0.1 * loss_dt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_train_loss = total_loss / len(dl_train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dl_val:
                pitch = batch['pitch']
                dt    = batch['dt']
                chord_e = chord_emb_model(batch['root'], batch['type'])
                logits, dt_pred, _ = model(pitch, dt, chord_e)
                loss_pitch = F.cross_entropy(logits.view(-1, logits.size(-1)), pitch.view(-1))
                loss_dt    = F.mse_loss(dt_pred, dt)
                val_loss += (loss_pitch + 0.1*loss_dt).item()
        avg_val_loss = val_loss / len(dl_val)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
