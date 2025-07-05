"""
Informer model implementation for wind and solar energy prediction.

This module implements the Informer model as described in the paper
"Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany".

The Informer model uses a probabilistic sparse self-attention mechanism to reduce
time complexity and memory usage for long sequence time-series forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np


class ProbSparseAttention(nn.Module):  
    """
    Probabilistic Sparse Self-Attention mechanism used in the Informer model.
    
    This attention mechanism selects the most informative queries based on
    the Kullback-Leibler (KL) divergence to reduce computation complexity.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        """
        Initialize the ProbSparse Attention.
        
        Args:
            mask_flag (bool): Whether to use masking for self-attention
            factor (int): Factor for controlling sparsity
            scale (float): Scaling factor for attention scores
            attention_dropout (float): Dropout rate for attention weights
        """
        super(ProbSparseAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Calculate the KL divergence-based sparsity measurement.
        
        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            sample_k (int): Number of samples to take
            n_top (int): Number of top queries to select
            
        Returns:
            torch.Tensor: Indices of top queries
        """
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        
        # Calculate max sample_k keys per query
        U_part = min(sample_k, L_K)
        
        # Sample keys for each query
        indices = torch.randperm(L_K)[:U_part]
        K_sample = K[:, :, indices, :]
        
        # Calculate KL divergence
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))
        
        # Find the top-u queries with largest KL divergence
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        return M_top
    
    def _get_initial_context(self, V, L_Q):
        """
        Initialize the context vector.
        
        Args:
            V (torch.Tensor): Value tensor
            L_Q (int): Length of query sequence
            
        Returns:
            torch.Tensor: Initial context vector
        """
        B, H, L_V, D = V.shape
        
        # Initialize with zeros
        context = torch.zeros((B, H, L_Q, D)).to(V.device)
        
        return context
    
    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass of the ProbSparse Attention.
        
        Args:
            queries (torch.Tensor): Query tensor
            keys (torch.Tensor): Key tensor
            values (torch.Tensor): Value tensor
            attn_mask (torch.Tensor): Attention mask
            
        Returns:
            tuple: (Output tensor, Attention weights)
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        # Reshape for multi-head attention
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Set scale factor
        scale = self.scale or 1./math.sqrt(D)
        
        # Calculate number of top queries
        n_top = max(int(L_Q * self.factor * math.log(L_K)), 1)
        
        # Get indices of top queries
        indices = self._prob_QK(queries, keys, n_top, n_top)
        
        # Initialize context
        context = self._get_initial_context(values, L_Q)
        
        # Calculate attention for top queries
        for i in range(B):
            for h in range(H):
                selected_Q = queries[i, h, indices[i, h], :]
                selected_Q = selected_Q.unsqueeze(0)
                
                # Calculate attention scores
                attn_scores = torch.matmul(selected_Q, keys[i, h].transpose(-2, -1)) * scale
                
                # Apply mask if needed
                if self.mask_flag and attn_mask is not None:
                    attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
                
                # Apply softmax
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Calculate context for selected queries
                selected_context = torch.matmul(attn_weights, values[i, h])
                
                # Update context
                context[i, h, indices[i, h], :] = selected_context
        
        # Reshape output
        context = context.transpose(1, 2)
        
        return context, None

class InformerEncoderLayer(nn.Module):
    """
    Encoder layer for the Informer model.
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initialize the Informer encoder layer.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward network
            dropout (float): Dropout rate
            activation (str): Activation function
        """
        super(InformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Multi-head attention
        self.self_attention = ProbSparseAttention(mask_flag=False, attention_dropout=dropout)
        self.multihead_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        """
        Forward pass of the encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention
        new_x, attn = self.multihead_attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed-forward network
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(y)
        x = self.norm2(x)
        
        return x

class InformerDecoderLayer(nn.Module):
    """
    Decoder layer for the Informer model.
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initialize the Informer decoder layer.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward network
            dropout (float): Dropout rate
            activation (str): Activation function
        """
        super(InformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-attention and cross-attention
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass of the decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
            memory (torch.Tensor): Memory from encoder
            tgt_mask (torch.Tensor): Target mask
            memory_mask (torch.Tensor): Memory mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention
        new_x, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Cross-attention
        new_x, _ = self.cross_attention(x, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        
        # Feed-forward network
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(y)
        x = self.norm3(x)
        
        return x

class InformerEncoder(nn.Module):
    """
    Encoder for the Informer model.
    """
    def __init__(self, encoder_layer, num_layers):
        """
        Initialize the Informer encoder.
        
        Args:
            encoder_layer (nn.Module): Encoder layer
            num_layers (int): Number of encoder layers
        """
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
    def forward(self, x, attn_mask=None):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            attn_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x

class InformerDecoder(nn.Module):
    """
    Decoder for the Informer model.
    """
    def __init__(self, decoder_layer, num_layers):
        """
        Initialize the Informer decoder.
        
        Args:
            decoder_layer (nn.Module): Decoder layer
            num_layers (int): Number of decoder layers
        """
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Input tensor
            memory (torch.Tensor): Memory from encoder
            tgt_mask (torch.Tensor): Target mask
            memory_mask (torch.Tensor): Memory mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Informer model.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class InformerModel(nn.Module):
    """
    Informer model for time series forecasting.
    
    As described in the paper, this model uses a probabilistic sparse self-attention
    mechanism to reduce time complexity and memory usage for long sequence forecasting.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, output_dim, dropout=0.1):
        """
        Initialize the Informer model.
        
        Args:
            input_dim (int): Number of input features
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Dimension of feed-forward network
            output_dim (int): Number of output features (forecast horizon)
            dropout (float): Dropout rate
        """
        super(InformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder
        encoder_layer = InformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = InformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = InformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = InformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
        # Store dimensions for later use
        self.d_model = d_model
        self.output_dim = output_dim
        
    def forward(self, src, tgt=None):
        """
        Forward pass through the network.
        
        Args:
            src (torch.Tensor): Source sequence of shape (batch_size, src_seq_len, input_dim)
            tgt (torch.Tensor, optional): Target sequence for teacher forcing
                                          Shape: (batch_size, tgt_seq_len, input_dim)
                                          
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Embed input sequence
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        
        # Encode source sequence
        memory = self.encoder(src)
        
        # Create target sequence if not provided (for inference)
        if tgt is None:
            # Use the last value of source as initial target
            tgt = src[:, -1:, :]
            
            # Generate target sequence autoregressively
            for i in range(self.output_dim - 1):
                # Add positional encoding to target
                tgt_with_pos = self.pos_encoder(tgt)
                
                # Create target mask to prevent attending to future positions
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1)).to(src.device)
                
                # Forward pass through decoder
                out = self.decoder(tgt_with_pos, memory, tgt_mask=tgt_mask)
                
                # Get the last prediction
                next_item = out[:, -1:, :]
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_item], dim=1)
        else:
            # Embed target sequence
            tgt = self.input_embedding(tgt)
            tgt = self.pos_encoder(tgt)
            
            # Create target mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1)).to(src.device)
            
            # Forward pass through decoder
            tgt = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Apply output layer to each time step
        output = self.output_layer(tgt).squeeze(-1)
        
        return output
