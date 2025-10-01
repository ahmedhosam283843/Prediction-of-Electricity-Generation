import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    """A learnable cyclic memory buffer.

    This module holds a learnable parameter tensor of shape (cycle_len, channel_size)
    and allows fetching a sequence of a given length starting from a specific index,
    wrapping around the cycle.
    """

    def __init__(self, cycle_len: int, channel_size: int):
        """
        Args:
            cycle_len (int): The length of the cycle.
            channel_size (int): The size of the feature dimension for each step in the cycle.
        """
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(
            cycle_len, channel_size), requires_grad=True)

    def forward(self, index: torch.Tensor, length: int) -> torch.Tensor:
        """
        Fetches a sequence of `length` from the cyclic buffer for each starting `index`.

        Args:
            index (torch.Tensor): A tensor of starting indices, shape (B, 1).
            length (int): The length of the sequence to fetch.

        Returns:
            torch.Tensor: The fetched sequences from the buffer, shape (B, length, channel_size).
        """
        # Create a range of offsets for the sequence length. Shape: (1, length)
        offsets = torch.arange(length, device=index.device).view(1, -1)
        # Calculate the indices to gather, wrapping around the cycle. Shape: (B, length)
        gather_index = (index.view(-1, 1) + torch.arange(length,
                        device=index.device).view(1, -1)) % self.cycle_len
        # Gather the data from the learnable buffer.
        return self.data[gather_index]


class CycleLSTMModel(nn.Module):
    """An LSTM model augmented with learnable cyclic patterns for seasonality.

    This model subtracts a learnable cyclic pattern from the input and adds another
    cyclic pattern to the output. This is designed to help the LSTM focus on
    non-seasonal variations in the time series.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, cycle_len: int, seq_len: int, dropout: float = 0.2):
        """
        Args:
            input_size (int): The number of features in the input sequence.
            hidden_size (int): The number of hidden units in the LSTM.
            num_layers (int): The number of LSTM layers.
            output_size (int): The forecast horizon (number of steps to predict).
            cycle_len (int): The length of the seasonal cycle (e.g., 24 for daily).
            seq_len (int): The length of the input sequence (lookback).
            dropout (float): The dropout probability for the LSTM.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Learnable cyclic pattern for input features
        self.input_cycle = RecurrentCycle(cycle_len, input_size)
        # Learnable cyclic pattern for the univariate output
        self.output_cycle = RecurrentCycle(cycle_len, 1)

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CycleLSTMModel.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, F).
            index (torch.Tensor): Starting index in the cycle for each sample, shape (B, 1).

        Returns:
            torch.Tensor: Output tensor of shape (B, H).
        """
        # Subtract the learnable seasonal pattern from the input.
        input_seasonal_pattern = self.input_cycle(index, self.seq_len)
        x_deseasonalized = x - input_seasonal_pattern

        # Initialize hidden and cell states for the LSTM.
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        # Process the deseasonalized sequence with the LSTM.
        lstm_out, _ = self.lstm(x_deseasonalized, (h0, c0))

        # Use the last time step's output for prediction.
        out = self.fc(lstm_out[:, -1, :])

        # Add the learnable seasonal pattern to the output.
        # The output pattern starts where the input pattern ended.
        output_start_index = (
            index + self.seq_len) % self.input_cycle.cycle_len
        output_seasonal_pattern = self.output_cycle(
            output_start_index, out.shape[1])
        out = out + output_seasonal_pattern.squeeze(2)

        return out
