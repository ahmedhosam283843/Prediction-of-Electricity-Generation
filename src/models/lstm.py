"""
LSTM model implementation for wind and solar energy prediction.

This module implements the Long Short-Term Memory (LSTM) model as described in the paper
"Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany".
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    
    As described in the paper, this model uses multiple LSTM layers with dropout
    and L2 regularization for improved performance and stability.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            output_dim (int): Number of output features (forecast horizon)
            dropout (float): Dropout probability
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state and cell state for bidirectional LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        # out shape: (batch_size, hidden_dim)
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        # out shape: (batch_size, output_dim)
        out = self.fc(out)
        
        return out