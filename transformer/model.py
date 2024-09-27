import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim, num_tickers):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_dim)

        # Set batch_first=True to improve inference performance
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=0.3,
            batch_first=True  # Ensures batch_first=True for better performance
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(hidden_dim, num_tickers)
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer

    def forward(self, src):
        src = self.embedding(src)  # Shape: (batch_size, time_steps, hidden_dim)
        src = self.transformer_encoder(src)  # Apply Transformer layers with batch_first=True
        src = self.dropout(src)  # Apply dropout
        src = src.mean(dim=1)  # Global average pooling over time steps
        output = self.fc_out(src)  # Output layer
        return output
