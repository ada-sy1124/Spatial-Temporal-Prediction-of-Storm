import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from livelossplot import PlotLosses
from tqdm.auto import tqdm



class BasicConv2d(nn.Module):
    """
    Standard convolutional block with Group Normalization and LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# RNNcell
class ConvRNNCell(nn.Module):
    """
    Simple Convolutional RNN cell computing hidden state transitions.
    """
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm=True):
        super().__init__()
        self.num_hidden = num_hidden
        padding = filter_size // 2

        # x → hidden
        self.conv_x = nn.Conv2d(
            in_channel, num_hidden, filter_size, stride, padding
        )

        # h → hidden
        self.conv_h = nn.Conv2d(
            num_hidden, num_hidden, filter_size, stride, padding
        )

        if layer_norm:
            self.ln = nn.LayerNorm([num_hidden, width, width])
        else:
            self.ln = nn.Identity()

    def forward(self, x, h_prev):
        # h_t = tanh(Wx * x + Wh * h_{t-1})
        h_new = self.conv_x(x) + self.conv_h(h_prev)
        h_new = self.ln(h_new)
        h_new = torch.tanh(h_new)
        return h_new


# ST-LSTMcell
class SpatioTemporalLSTMCell(nn.Module):
    """
    Spatio-Temporal LSTM cell for PredRNN.
    Manages dual memory states: standard temporal (C) and spatial (M)
    This structure come from the paper:
    **Wang Y, Long M, Wang J, et al. Predrnn: Recurrent neural networks
    for predictive learning using
    spatiotemporal lstms[J]. Advances in neural information processing systems,
    2017, 30.**
    """
    def __init__(
            self, in_channel, num_hidden, width,
            filter_size, stride, layer_norm=True):
        super().__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        # Gates for input (i, f, g, o) and spatial (i', f', g')
        self.conv_x = nn.Conv2d(
            in_channel, num_hidden * 7, filter_size, stride, self.padding)

        # Temporal hidden state gates
        self.conv_h = nn.Conv2d(
            num_hidden, num_hidden * 4, filter_size, stride, self.padding)

        # Spatial memory gates
        self.conv_m = nn.Conv2d(
            num_hidden, num_hidden * 3, filter_size, stride, self.padding)

        if layer_norm:
            self.ln_x = nn.LayerNorm([num_hidden * 7, width, width])
            self.ln_h = nn.LayerNorm([num_hidden * 4, width, width])
            self.ln_m = nn.LayerNorm([num_hidden * 3, width, width])
        else:
            self.ln_x = nn.Identity()
            self.ln_h = nn.Identity()
            self.ln_m = nn.Identity()

        # Final fusion of temporal and spatial memory
        self.conv_cm = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x, h_t, c_t, m_t):

        x_concat = self.ln_x(self.conv_x(x))
        h_concat = self.ln_h(self.conv_h(h_t))
        m_concat = self.ln_m(self.conv_m(m_t))

        # Split concatenated features into specific gate activations
        i_x, f_x, g_x, o_x, i_xm, f_xm, g_xm = torch.chunk(x_concat, 7, dim=1)

        i_h, f_h, g_h, o_h = torch.chunk(h_concat, 4, dim=1)

        i_m, f_m, g_m = torch.chunk(m_concat, 3, dim=1)

        # Temporal cell state update
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self._forget_bias)
        g = torch.tanh(g_x + g_h)
        c_new = f * c_t + i * g

        # Spatiotemporal memory state update (M)
        i2 = torch.sigmoid(i_xm + i_m)
        f2 = torch.sigmoid(f_xm + f_m + self._forget_bias)
        g2 = torch.tanh(g_xm + g_m)
        m_new = f2 * m_t + i2 * g2

        # Output gate and hidden state fusion
        o = torch.sigmoid(o_x + o_h)
        cm = torch.cat([c_new, m_new], dim=1)          # (B, 2H, W, W)
        h_hat = torch.tanh(self.conv_cm(cm))           # (B, H, W, W)
        h_new = o * h_hat

        return h_new, c_new, m_new


# (C) Encoder: 384x384 -> 48x48
class Encoder(nn.Module):
    """
    Downsamples high-resolution input frames into compressed feature maps.
    """
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.enc = nn.Sequential(
            BasicConv2d(in_channels, 32, 3, 2, 1),           # 384 -> 192
            BasicConv2d(32, 64, 3, 2, 1),                    # 192 -> 96
            BasicConv2d(64, hid_channels, 3, 2, 1)           # 96 -> 48
        )

    def forward(self, x):
        return self.enc(x)


# (D) Decoder: 48x48 -> 384x384
class Decoder(nn.Module):
    """
    Upsamples latent features back to original frame dimensions using bilinear interpolation.
    """
    def __init__(self, hid_channels, out_channels):
        super().__init__()
        self.dec = nn.Sequential(
            # 48 -> 96
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(hid_channels, 64, 3, 1, 1),

            # 96 -> 192
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(64, 32, 3, 1, 1),

            # 192 -> 384
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(32, 16, 3, 1, 1),

            # Final convolution to reach target channel depth
            nn.Conv2d(16, out_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dec(x)


class PredRNN(nn.Module):
    """
    PredRNN architecture implementation using 
    stacked SpatioTemporal LSTM (ST-LSTM) cells.

    PredRNN improves upon standard ConvLSTMs by introducing a
    shared spatial memory (M) that bypasses the temporal state.
    This allows the model to communicate spatial information vertically
    across layers and horizontally across time steps in a 'zig-zag' pattern.

    Attributes:
        hidden_dim (int): Number of feature channels in the hidden states.
        num_layers (int): Number of ST-LSTM layers stacked vertically.
        encoder (nn.Module): Downsamples high-res input frames to
                            latent space (48x48).
        decoder (nn.Module): Upsamples latent representations back to
                            original resolution.
        cell_list (nn.ModuleList): Collection of ST-LSTM cells for
                                    sequential processing.
    """
    def __init__(self, in_channels=1, hidden_dim=64, layers=3):
        super(PredRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.frame_channel = hidden_dim

        # Mapping input space (e.g., 384x384) to latent space (e.g., 48x48)
        self.encoder = Encoder(in_channels, hidden_dim)
        # Mapping latent space back to output space
        self.decoder = Decoder(hidden_dim, in_channels)

        cells = []
        for i in range(self.num_layers):
            # ST-LSTM Cell requires hidden_dim, spatial memory dim, 
            # and kernel specs
            cells.append(SpatioTemporalLSTMCell(self.frame_channel, hidden_dim, 48, 3, 1, True))
        self.cell_list = nn.ModuleList(cells)

    def forward(self, x, input_len=12, total_len=24):
        """
        Forward pass for PredRNN.
        
        Args:
            x (Tensor): Input sequence of shape (B, T_in, C, H, W).
            input_len (int): Number of context frames provided (observation window).
            total_len (int): Total sequence length (input_len + prediction_len).

        Returns:
            output (Tensor): Predicted frames of shape (B, T_pred, C, H, W).
        """
        # x shape: (Batch, Time, Channel, Height, Width)
        b, t, c, h, w = x.shape

        # h_t and c_t store the hidden and cell states for EACH layer at the current time t
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            h_t.append(torch.zeros(b, self.hidden_dim, 48, 48).to(x.device))
            c_t.append(torch.zeros(b, self.hidden_dim, 48, 48).to(x.device))

        # m_t is the Spatio-Temporal Memory. Unlike h_t, it is SHARED across all layers 
        # within a single time step, effectively flowing from layer i to layer i+1.
        m_t = torch.zeros(b, self.hidden_dim, 48, 48).to(x.device)

        gen_frames = []

        for t in range(total_len):
            # 1. ENCODING/INPUT STAGE
            if t < input_len:
                # Use ground truth frames for the input phase
                net = self.encoder(x[:, t])
            else:
                # Use the top-layer hidden state from the previous step for autoregression
                net = h_t[-1]

            # 2. RECURRENT TRANSITION STAGE
            for i in range(self.num_layers):
                # The ST-LSTM updates h, c (temporal) AND m (spatial)
                # m_t flows through this loop, updating layer by layer
                h_t[i], c_t[i], m_t = self.cell_list[i](net, h_t[i], c_t[i], m_t)
                net = h_t[i] # Output of layer i becomes input to layer i+1

            # 3. DECODING/GENERATION STAGE
            if t >= input_len:
                # Only decode frames for the prediction horizon
                x_gen = self.decoder(net)
                gen_frames.append(x_gen)

        # Stack predictions along the time dimension (dim=1)
        output = torch.stack(gen_frames, dim=1)
        return output

    
class BaselineCNN(nn.Module):
    """
    Baseline CNN for spatio-temporal prediction using autoregressive feedback.
    
    This model serves as a non-recurrent baseline. Unlike ConvRNN, it has no 
    internal memory (hidden state). It predicts the next frame based solely 
    on the spatial features of the previous frame, feeding its own 
    predictions back as input during the inference phase.

    Attributes:
        encoder (nn.Module): Compresses input pixels into latent feature maps.
        decoder (nn.Module): Reconstructs pixels from latent feature maps.
        main_conv (nn.Sequential): Spatial transition layers that map current 
                                   features to predicted future features.
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = Encoder(in_channels, hidden_dim)
        self.decoder = Decoder(hidden_dim, in_channels)

        # Spatial transformation block to project features to the next timestep
        self.main_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

    def forward(self, x, input_len=12, total_len=24):
        gen_frames = []
        last_prediction = None

        for t in range(total_len):
            # Phase 1: Process the input sequence (Teacher Forcing/Warm-up)
            if t < input_len:
                # Encode ground truth frame
                latent = self.encoder(x[:, t])
                # Predict next state's latent features
                pred_latent = self.main_conv(latent)
                # Generate pixel-space prediction
                x_gen = self.decoder(pred_latent)
                # Store prediction to use as input for the next step
                last_prediction = x_gen
            
            # Phase 2: Recursive prediction (Autoregressive)
            else:
                # Encode the model's own previous prediction
                latent = self.encoder(last_prediction)
                # Predict next state's latent features
                pred_latent = self.main_conv(latent)
                # Generate pixel-space prediction
                x_gen = self.decoder(pred_latent)
                
                # Collect output frames for the future sequence
                gen_frames.append(x_gen)
                # Feedback loop: update the input for the next timestep
                last_prediction = x_gen

        # Return the stacked tensor of predicted future frames
        return torch.stack(gen_frames, dim=1)



