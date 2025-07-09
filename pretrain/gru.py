"""
File: gru.py
Description: GRU model for inference.
File Modified: 20/06/2025
Python Version: 3.9
"""

# Imports - Only PyTorch is needed now
import torch
import torch.nn as nn

# GRU Model inheriting from the standard torch.nn.Module
class GRU(nn.Module):
    """
    Constructor for the inference model.
    Training-specific parameters like learning rate (lr) have been removed.
    """
    def __init__(self, n_features=4, hidden_units=100, n_layers=10):
        super(GRU, self).__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        # The actual model architecture remains the same
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(self.hidden_units, 1)

    # Forward Pass - This is the core function needed for prediction
    def forward(self, x):
        # Create the initial hidden state on the same device as the input tensor 'x'
        # This makes the model flexible to run on CPU or GPU without code changes.
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units, device=x.device)

        # We detach h0 to prevent gradients from flowing back into it, which is good practice.
        out, _ = self.gru(x, h0.detach())

        # Get the output from the last time step and pass it to the linear layer
        out = self.linear(out[:, -1, :])
        return out

    # All training-related methods (training_step, validation_step, configure_optimizers, etc.)
    # have been removed as they are not needed for inference.