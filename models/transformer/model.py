import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_dim):
        """
        Initializes the Transformer model for stock price prediction for a single ticker.

        Args:
            input_size (int): Number of input features.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            hidden_dim (int): Size of the hidden dimension.
        """
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
        self.fc_out = nn.Linear(hidden_dim, 1)  # Single output for regression
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer

    def forward(self, src):
        """
        Forward pass for the model.

        Args:
            src (torch.Tensor): Input tensor with shape (batch_size, time_steps, input_size).

        Returns:
            torch.Tensor: A single scalar prediction per batch for the regression task.
        """
        src = self.embedding(src)  # Shape: (batch_size, time_steps, hidden_dim)
        src = self.transformer_encoder(src)  # Apply Transformer layers
        src = self.dropout(src)  # Apply dropout
        src = src.mean(dim=1)  # Global average pooling over time steps to reduce to (batch_size, hidden_dim)
        output = self.fc_out(src)  # Single value output (batch_size, 1)
        return output
