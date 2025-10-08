import torch
import torch.nn as nn

class RecurrentCycle(nn.Module):
    """
    Learnable cyclical bias table indexed by (hour-of-cycle, ...).

    Stores a (cycle_len, channel_size) parameter matrix and returns a rolling
    window of length `length` starting at `index` (mod cycle_len).
    """

    def __init__(self, cycle_len, channel_size):
        """
        Args:
            cycle_len (int): Length of the repeating cycle (e.g., 24 for hours).
            channel_size (int): Number of channels/columns to retrieve per step.
        """
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        """
        Args:
            index (Tensor): (batch, 1) integer start indices in [0, cycle_len).
            length (int): Length of the rolling window to gather.

        Returns:
            Tensor: (batch, length, channel_size)
        """
        # Build indices for a contiguous window modulo cycle length
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]

class CycleLSTMModel(nn.Module):
    """
    LSTM forecaster with learnable cyclical offsets on inputs/outputs.

    - Subtracts a cyclical bias from inputs (to remove seasonal pattern).
    - Runs an LSTM over the de-seasonalized inputs.
    - Adds a cyclical bias back to the outputs at the appropriate future indices.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, cycle_len, seq_len, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): LSTM hidden size.
            num_layers (int): Number of LSTM layers.
            output_size (int): Forecast horizon (steps ahead).
            cycle_len (int): Length of the seasonal cycle (e.g., 24).
            seq_len (int): Input sequence length.
            dropout (float): Dropout in LSTM between layers.
        """
        super(CycleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_size = output_size
        self.cycle_len = cycle_len

        # Cyclical tables for input (all features) and output (univariate)
        self.cycleQueue_input = RecurrentCycle(cycle_len, input_size)
        self.cycleQueue_output = RecurrentCycle(cycle_len, 1)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, index):
        """
        Forward pass.

        Args:
            x (Tensor): (batch_size, seq_len, input_size)
            index (Tensor): (batch_size, 1) cycle indices for the first input step

        Returns:
            Tensor: (batch_size, output_size)
        """
        # Remove cyclical bias from inputs
        cq_input = self.cycleQueue_input(index, self.seq_len)  # (batch_size, seq_len, input_size)
        x = x - cq_input

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM encoder
        out, _ = self.lstm(x, (h0, c0))             # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])                # (batch_size, output_size)

        # Add cyclical bias for predicted steps: start from (index + seq_len)
        cp_output = self.cycleQueue_output((index + self.seq_len) % self.cycle_len, self.output_size)  # (batch, output_size, 1)
        out = out + cp_output.squeeze(2)            # (batch_size, output_size)
        return out