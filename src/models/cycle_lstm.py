import torch
import torch.nn as nn

class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]

class CycleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cycle_len, seq_len, dropout=0.2):
        super(CycleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_size = output_size
        self.cycle_len = cycle_len
        self.cycleQueue_input = RecurrentCycle(cycle_len, input_size)  # Cycle for all input features
        self.cycleQueue_output = RecurrentCycle(cycle_len, 1)  # Cycle for univariate output
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, index):
        # x: (batch_size, seq_len, input_size)
        # index: (batch_size, 1)
        cq_input = self.cycleQueue_input(index, self.seq_len)  # (batch_size, seq_len, input_size)
        x = x - cq_input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # (batch_size, output_size)
        cp_output = self.cycleQueue_output((index + self.seq_len) % self.cycle_len, self.output_size)  # (batch_size, output_size, 1)
        out = out + cp_output.squeeze(2)  # (batch_size, output_size)
        return out