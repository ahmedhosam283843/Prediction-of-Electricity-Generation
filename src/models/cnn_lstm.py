import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        # LSTM layers
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # To (batch_size, input_dim, seq_len) for Conv1d
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, 128)
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last timestep
        out = self.fc(out)
        return out