import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy.signal import savgol_coeffs
from Smoothing.smoother import Smoother


# An interface / Abstract class for smoothing operations. 
# This is used outside the inference loop, as a last effort to smooth the movement data.
class SavGolKernelSmootherGPU(Smoother):
    def __init__(self, 
                 window_size = 7,           # The size of the window for the SAVGOL filter. Must be odd.
                 poly_order = 3,            # The order of the polynomial to fit.
                 smoothing_factor = 0.5,    # Device to run computations on ('cuda', 'cpu', or specific device)
                 device=None):
        super().__init__(smoothing_factor)
        
        # Determine device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        self.window_size = window_size
        self.poly_order = poly_order
        
        # Generate the SAVGOL kernel
        kernel = savgol_coeffs(window_size, poly_order)
        
        # Create the conv1d layer with fixed weights
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=window_size,
            padding=window_size // 2,
            padding_mode='replicate',
            bias=False
        ).to(self.device)
        
        # Set the weights to the SAVGOL kernel and make them non-learnable
        kernel_tensor = torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)
        self.conv.weight = nn.Parameter(kernel_tensor, requires_grad=False)
    

    def smooth_sequence(self, 
                        data: Tensor,   # Tensor of shape [sequence_length, feature_dim]
    ) -> Tensor:                        # Rreturn: Smoothed data of the same shape
        
        # Ensure data is on the correct device
        data = data.to(self.device)

        # Becouse we might use the original tensor in the continues infurrence process, 
        # we clone it here, so that we do not modify the original data.
        result = data.clone()
        seq_len, feature_dim = data.shape
        
        # Process each feature dimension with the convolutional filter
        for dim in range(feature_dim):
            # Extract feature sequence and reshape for conv1d [batch, channels, seq_len]
            feature_seq = data[:, dim].view(1, 1, -1)
            
            # Apply convolution (the kernel)
            smoothed_seq = self.conv(feature_seq)
            
            # Reshape back and apply smoothing factor
            smoothed_seq = smoothed_seq.view(-1)
            original_seq = data[:, dim]
            
            # Blend original and smoothed based on smoothing factor
            result[:, dim] = (1 - self.smoothing_factor) * original_seq + self.smoothing_factor * smoothed_seq
        
        return result
    

    def smooth_frame(self, 
                     data: Tensor,      # Tensor of shape [sequence_length, feature_dim]
                     frame_index: int   # Index of the frame to smooth
        ):

        # For efficiency, we smooth the whole sequence and extract the frame
        # This is actually more efficient on GPU than setting up just for one frame
        smoothed_sequence = self.smooth_sequence(data)
        return smoothed_sequence[frame_index]
    
    def to(self, device):
        self.device = device
        self.conv = self.conv.to(device)
        return self