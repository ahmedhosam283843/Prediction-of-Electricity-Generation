"""
Integration with Non-Stationary Transformer repository for wind and solar energy prediction.

This module provides an interface to the Non-Stationary Transformer model as described in the paper
"Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany".

The Non-Stationary Transformer model addresses the non-stationarity in time series data
by using de-stationary attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

class NonStationaryTransformerWrapper(nn.Module):
    """
    Wrapper for the Non-Stationary Transformer model from the official repository.
    
    As mentioned in the paper, this implementation uses the official code from:
    https://github.com/thuml/Nonstationary_Transformers
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, output_dim, 
                 stochastic=True, dropout=0.1):
        """
        Initialize the Non-Stationary Transformer wrapper.
        
        Args:
            input_dim (int): Number of input features
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Dimension of feed-forward network
            output_dim (int): Number of output features (forecast horizon)
            stochastic (bool): Whether to use stochastic or deterministic de-stationary factors
            dropout (float): Dropout rate
        """
        super(NonStationaryTransformerWrapper, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.stochastic = stochastic
        self.dropout = dropout
        
        # Flag to check if the model is initialized
        self.is_initialized = False
        
    def initialize_model(self):
        """
        Initialize the Non-Stationary Transformer model from the official repository.
        
        This method should be called before using the model to ensure
        the repository is cloned and the model is properly initialized.
        """
        # Check if the model is already initialized
        if self.is_initialized:
            return
        
        # Clone the repository if it doesn't exist
        repo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'external', 'Nonstationary_Transformers')
        if not os.path.exists(repo_dir):
            os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
            os.system(f'git clone https://github.com/thuml/Nonstationary_Transformers.git {repo_dir}')
        
        # Add the repository to the Python path
        sys.path.append(repo_dir)
        
        # Import the Non-Stationary Transformer model
        try:
            from models.ns_transformer import Model
            
            # Define model arguments
            self.args = type('Args', (), {
                'enc_in': self.input_dim,
                'dec_in': self.input_dim,
                'c_out': 1,
                'd_model': self.d_model,
                'n_heads': self.nhead,
                'e_layers': self.num_encoder_layers,
                'd_layers': self.num_decoder_layers,
                'd_ff': self.dim_feedforward,
                'dropout': self.dropout,
                'embed': 'timeF',
                'freq': 'h',
                'activation': 'gelu',
                'output_attention': False,
                'distil': True,
                'moving_avg': 25,
                'factor': 1,
                'L': self.output_dim,
                'kernel_size': 25,
                'stochastic': self.stochastic
            })
            
            # Initialize the model
            self.model = Model(self.args)
            
            # Set the flag
            self.is_initialized = True
            
        except ImportError as e:
            print(f"Error importing Non-Stationary Transformer: {e}")
            print("Please make sure the Nonstationary_Transformers repository is properly cloned.")
            raise
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass through the network.
        
        Args:
            x_enc (torch.Tensor): Encoder input of shape (batch_size, seq_len, input_dim)
            x_mark_enc (torch.Tensor, optional): Encoder time features
            x_dec (torch.Tensor, optional): Decoder input
            x_mark_dec (torch.Tensor, optional): Decoder time features
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize the model if not already done
        if not self.is_initialized:
            self.initialize_model()
        
        # If decoder inputs are not provided, create them
        if x_dec is None:
            # Use the last value of encoder input as initial decoder input
            x_dec = x_enc[:, -self.output_dim:, :]
            
            # If time features are provided, use them for decoder as well
            if x_mark_enc is not None:
                x_mark_dec = x_mark_enc[:, -self.output_dim:, :]
        
        # Forward pass through the Non-Stationary Transformer model
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        return output

def create_ns_transformer_model(config):
    """
    Create a Non-Stationary Transformer model with the given configuration.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        NonStationaryTransformerWrapper: Initialized Non-Stationary Transformer model
    """
    model = NonStationaryTransformerWrapper(
        input_dim=config.get('input_dim', 1),
        d_model=config.get('TRANSFORMER_D_MODEL', 512),
        nhead=config.get('TRANSFORMER_N_HEADS', 8),
        num_encoder_layers=config.get('TRANSFORMER_NUM_ENCODER_LAYERS', 3),
        num_decoder_layers=config.get('TRANSFORMER_NUM_DECODER_LAYERS', 3),
        dim_feedforward=config.get('TRANSFORMER_DIM_FEEDFORWARD', 2048),
        output_dim=config.get('FORECAST_HORIZON', 24),
        stochastic=config.get('NS_TRANSFORMER_STOCHASTIC', True),
        dropout=config.get('TRANSFORMER_DROPOUT', 0.1)
    )
    
    return model
