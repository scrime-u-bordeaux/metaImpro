import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, dim=128, layers=1):
        super().__init__()
        self.dim = dim
        self.layers = layers