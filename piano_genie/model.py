import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from typing import Dict, Optional, Tuple

class IQAE_Quantizer(nn.Module):
    """
    Integer-Quantized AutoEncoder Quantizer
    Quantifie les valeurs scalaires vers `num_buckets` bins uniformément espacés dans [-1, 1]
    Applique un estimateur straight-through pour le passage des gradients.
    """
    buckets: torch.Tensor

    def __init__(self, num_buckets: int = 8):
        super().__init__()
        self.num_buckets = num_buckets
        self.register_buffer('buckets', torch.linspace(-1.0, 1.0, num_buckets))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_clamped = torch.clamp(x, -1.0, 1.0)
        distances = torch.abs(x_clamped.unsqueeze(-1) - self.buckets.view(1, 1, -1))
        quantized_indices = torch.argmin(distances, dim=-1)
        quantized_values = self.buckets[quantized_indices]
        quantized_values = x_clamped + (quantized_values - x_clamped).detach()
        return quantized_indices, quantized_values

class Encoder(nn.Module):
    """
    Piano Genie Encoder: Bidirectional LSTM + Quantizer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        proj_size: int,
        num_buckets: int = 8
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, proj_size)
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_size * 2, 1)
        self.quantizer = IQAE_Quantizer(num_buckets)

    def forward(
        self,
        features: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(features)
        packed = rnn_utils.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = rnn_utils.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=features.size(1)
        )
        continuous_values = self.output_proj(outputs).squeeze(-1)
        quantized_indices, quantized_values = self.quantizer(continuous_values)
        return quantized_indices, quantized_values, continuous_values

class Decoder(nn.Module):
    """
    Piano Genie Decoder: Unidirectional LSTM
    """
    def __init__(
        self,
        num_buckets: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 89
    ):
        super().__init__()
        self.bucket_embedding = nn.Embedding(num_buckets, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        bucket_indices: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> torch.Tensor:
        x = self.bucket_embedding(bucket_indices.long())
        packed = rnn_utils.pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = rnn_utils.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=bucket_indices.size(1)
        )
        logits = self.output_proj(outputs)
        return logits

class PianoGenie(nn.Module):
    """
    Modèle Piano Genie complet avec IQAE
    """
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        proj_size: int = 128,
        num_buckets: int = 8,
        output_size: int = 89
    ):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, proj_size, num_buckets)
        self.decoder = Decoder(num_buckets, hidden_size, num_layers, output_size)

    def forward(
        self,
        features: torch.Tensor,
        seq_lens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        bucket_indices, quantized_values, continuous_values = self.encoder(features, seq_lens)
        logits = self.decoder(bucket_indices, seq_lens)
        probabilities = F.softmax(logits, dim=-1)
        return {
            'bucket_indices': bucket_indices,
            'quantized_values': quantized_values,
            'continuous_values': continuous_values,
            'logits': logits,
            'probabilities': probabilities
        }


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    seq_lens: torch.Tensor,
    contour_weight: float = 1.0,
    margin_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    B, T = targets.shape
    device = targets.device
    mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)

    logits = outputs['logits']
    loss_recons = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    ).reshape(B, T)
    loss_recons = (loss_recons * mask).sum() / mask.sum()

    continuous = outputs['continuous_values']
    margin_viol = torch.clamp(torch.abs(continuous) - 1, min=0) ** 2
    loss_margin = (margin_viol * mask).sum() / mask.sum()

    loss_contour = torch.tensor(0.0, device=device)
    if T > 1:
        tgt_diff = targets[:, 1:] - targets[:, :-1]
        enc_diff = continuous[:, 1:] - continuous[:, :-1]
        diff_mask = mask[:, 1:]
        contour_viol = torch.clamp(1 - tgt_diff.sign() * enc_diff.sign(), min=0) ** 2
        loss_contour = (contour_viol * diff_mask).sum() / diff_mask.sum()

    total_loss = loss_recons + margin_weight * loss_margin + contour_weight * loss_contour
    return {
        'total_loss': total_loss,
        'reconstruction_loss': loss_recons,
        'margin_loss': loss_margin,
        'contour_loss': loss_contour
    }


def build_model_and_features(
    pitches: torch.Tensor,
    batch_size: int,
    seq_len: int,
    seq_varlens: Optional[torch.Tensor] = None,
    is_training: bool = True
) -> Tuple[PianoGenie, Dict[str, torch.Tensor]]:
    """
    Normalise les pitches [0,88] vers [-1,1], construit un batch de séquences,
    et renvoie le modèle + features dict.
    """
    # pitches: (B, T) valeurs entre 0 et 88 inclus
    pitch_norm = (pitches.float() / 88.0) * 2.0 - 1.0
    features = pitch_norm.unsqueeze(-1)

    # Mode entaînement
    if is_training:
        seq_lens = torch.randint(32, seq_len + 1, (batch_size,))

    # Lorsque la longueur du tenseur est explicite
    elif seq_varlens is not None:
        seq_lens = seq_varlens
    
    # vecteurs tous de taille seq_len
    else:
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)

    model = PianoGenie()
    return model, {'features': features, 'seq_lens': seq_lens, 'targets': pitches}

"""
if __name__ == '__main__':
    B, T = 4, 128 # B: batch_size, T: Sequence_length
    pitches = torch.randint(0, 89, (B, T))  # 0 à 88 inclus
    model, feat = build_model_and_features(pitches, B, T)
    out = model(feat['features'], feat['seq_lens'])
    losses = compute_losses(out, feat['targets'], feat['seq_lens'])
    print(f"Bucket indices: {out['bucket_indices'].shape}")
    print(f"Logits: {out['logits'].shape}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
"""