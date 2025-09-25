import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super(CNNLSTMModel, self).__init__()
        # Convolutional layers (smaller)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        # LSTM layer (smaller hidden size)
        self.lstm = nn.LSTM(64, 256, num_layers=1, batch_first=True,
                            dropout=dropout, bidirectional=False)
        # Fully connected layers (smaller)
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # To (batch, input_dim, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Back to (batch, seq_len, 64)
        out, _ = self.lstm(x)  # out: (batch, seq_len, 256)
        out = out[:, -1, :]  # Take last timestep: (batch, 256)
        out = torch.relu(self.fc1(out))  # (batch, 256)
        out = self.bn_fc(out)  # (batch, 256)
        out = self.fc_dropout(out)
        out = self.fc2(out)  # (batch, output_dim)
        return out
