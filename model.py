# Voir https://colab.research.google.com/drive/124pk1yehPx1y-K3hBG6-SoUSVqQ-RWnM?usp=sharing#scrollTo=9UQ9PAvMCwd2
import torch
import torch.nn as nn
import torch.nn.functional as F

PIANO_LOWEST_KEY_MIDI_PITCH = 21
PIANO_NUM_KEYS = 88

SOS = PIANO_NUM_KEYS


# Decoder
class PianoGenieDecoder(nn.Module):
    def __init__(self, rnn_dim=128, rnn_num_layers=2):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.rnn_num_layers = rnn_num_layers
        self.input = nn.Linear(PIANO_NUM_KEYS + 3, rnn_dim)
        self.lstm = nn.LSTM(
            rnn_dim,
            rnn_dim,
            rnn_num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output = nn.Linear(rnn_dim, 88)
    
    def init_hidden(self, batch_size, device=None):
        h = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_dim, device=device)
        c = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_dim, device=device)
        return (h, c)

    def forward(self, k, t, b, h_0=None):
        # Prepend <S> token to shift k_i to k_{i-1}
        k_m1 = torch.cat([torch.full_like(k[:, :1], SOS), k[:, :-1]], dim=1)

        # Encode input
        inputs = [
            F.one_hot(k_m1, PIANO_NUM_KEYS + 1),
            t.unsqueeze(dim=2),
            b.unsqueeze(dim=2),
        ]
        x = torch.cat(inputs, dim=2)

        # Project encoded inputs
        x = self.input(x)

        # Run RNN
        if h_0 is None:
            h_0 = self.init_hidden(k.shape[0], device=k.device)
        x, h_N = self.lstm(x, h_0)

        # Compute logits
        hat_k = self.output(x)

        return hat_k, h_N

# Encoder
class PianoGenieEncoder(nn.Module):
    def __init__(self, rnn_dim=128, rnn_num_layers=2):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.rnn_num_layers = rnn_num_layers
        self.input = nn.Linear(PIANO_NUM_KEYS + 1, rnn_dim)
        self.lstm = nn.LSTM(
            rnn_dim,
            rnn_dim,
            rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(rnn_dim * 2, 1)

    def forward(self, k, t):
        inputs = [
            F.one_hot(k, PIANO_NUM_KEYS),
            t.unsqueeze(dim=2),
        ]
        x = self.input(torch.cat(inputs, dim=2))
        # NOTE: PyTorch uses zeros automatically if h is None
        x, _ = self.lstm(x, None)
        x = self.output(x)
        return x[:, :, 0]

# IQAE quantizer
class IntegerQuantizer(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins

    def real_to_discrete(self, x, eps=1e-6):
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        x *= self.num_bins - 1
        x = (torch.round(x) + eps).long()
        return x

    def discrete_to_real(self, x):
        x = x.float()
        x /= self.num_bins - 1
        x = (x * 2) - 1
        return x

    def forward(self, x):
        # Quantize and compute delta (used for straight-through estimator)
        with torch.no_grad():
            x_disc = self.real_to_discrete(x)
            x_quant = self.discrete_to_real(x_disc)
            x_quant_delta = x_quant - x

        # @markdown In the backwards pass, we will use the straight-through estimator (Bengio et al. 2013), i.e., pretend that this discretization did not happen when computing gradients.
        # Quantize w/ straight-through estimator
        x = x + x_quant_delta

        return x


# Autoencoder : rénuion 
class PianoGenieAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = PianoGenieEncoder(
            rnn_dim=cfg["model_rnn_dim"],
            rnn_num_layers=cfg["model_rnn_num_layers"],
        )
        self.quant = IntegerQuantizer(cfg["num_buttons"])
        self.dec = PianoGenieDecoder(
            rnn_dim=cfg["model_rnn_dim"],
            rnn_num_layers=cfg["model_rnn_num_layers"],
        )

    def forward(self, k, t):
        e = self.enc(k, t)
        b = self.quant(e)
        hat_k, _ = self.dec(k, t, b)
        return hat_k, e