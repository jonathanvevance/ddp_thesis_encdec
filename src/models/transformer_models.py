"""Python file with MLP model classes."""

import math
import torch
import torch.nn as nn

from utils.vocab_utils import PAD_INDEX

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransDecoder(nn.Module):

    def __init__(self, d_model, n_head, num_dec_layers, vocab_size, device):
        super().__init__()
        self.tgt_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx = PAD_INDEX, device = device
        )
        self.pos_encoder = PositionalEncoding(d_model = d_model)

        dec_layers = nn.TransformerDecoderLayer(d_model = d_model, nhead = n_head)
        self.decoder = nn.TransformerDecoder(dec_layers, num_layers = num_dec_layers)

    def forward(self, target, target_padding_mask, memory, memory_padding_mask):
        abc = self.tgt_embedding(target)
        print(abc.shape)
        pass
