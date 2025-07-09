# simple_lstm.py
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleLSTM(nn.Module):
    def __init__(self, n_features=7, hidden_units=100, n_layers=10):
        super(SimpleLSTM, self).__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1
        )
        self.linear = nn.Linear(hidden_units, 1)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_units).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
