# transformer_model.py
import torch
import torch.nn as nn
from torch.nn import Transformer

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim, output_dim, dropout=0.1):
        """
        Transformer model that processes multiple time-series features (OHLCV).
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)  # Input will be 5 (OHLCV)
        self.transformer = Transformer(
            d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src)  # Ensure input matches the d_model (hidden_dim)
        src = self.transformer(src, src)
        out = self.fc_out(src[:, -1, :])  # Predict based on the last output of the sequence
        return out
