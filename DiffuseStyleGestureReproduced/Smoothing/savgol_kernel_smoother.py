
import torch
from torch import Tensor
import numpy as np
from scipy.signal import savgol_coeffs
from Smoothing.smoother import Smoother

# An interface / Abstract class for smoothing operations. 
# This is used outside the inference loop, as a last effort to smooth the movement data.
class SavGolKernelSmoother(Smoother):
    def __init__(self, 
                window_size = 7,        # The size of the window for the SAVGOL filter. Must be odd.
                poly_order = 3,         # The order of the polynomial to fit.
                smoothing_factor = 0.5  # Smoothing factor to blend original and smoothed data
    ):
        super().__init__(smoothing_factor)
        
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        self.window_size = window_size
        self.poly_order = poly_order
        
        # Generate the SAVGOL kernel
        self.kernel = torch.tensor(savgol_coeffs(window_size, poly_order), dtype=torch.float32)
    
    def smooth_sequence(self, 
                        data: Tensor    # Tensor of shape [sequence_length, feature_dim]
    ) -> Tensor:                    # Smoothed data of the same shape
        
        result = data.clone()
        seq_len, feature_dim = data.shape
        half_window = self.window_size // 2
        kernel = self.kernel.cpu().numpy()
        
        # Process each feature dimension
        for dim in range(feature_dim):
            feature_seq = data[:, dim].cpu().numpy()
            smoothed_seq = np.zeros_like(feature_seq)
            
            # For each frame in the sequence
            for i in range(seq_len):
                # Define window boundaries
                left = max(0, i - half_window)
                right = min(seq_len, i + half_window + 1)
                window = feature_seq[left:right]
                
                # Pad window if needed (near sequence boundaries)
                if len(window) < self.window_size:
                    if i < half_window:  # Near start
                        pad_left = half_window - i
                        window = np.pad(window, (pad_left, 0), mode='edge')
                    else:  # Near end
                        pad_right = self.window_size - len(window)
                        window = np.pad(window, (0, pad_right), mode='edge')
                
                # Apply kernel
                smoothed_seq[i] = np.sum(window * kernel)
            
            # Blend original and smoothed based on smoothing factor
            blended = (1 - self.smoothing_factor) * feature_seq + self.smoothing_factor * smoothed_seq
            result[:, dim] = torch.tensor(blended, device=data.device, dtype=data.dtype)
        
        return result
    
    def smooth_frame(self, 
                     data: Tensor,      # Tensor of shape [sequence_length, feature_dim]
                     frame_index: int   # Index of the frame to smooth (0-based index, must be within sequence length range
    ) -> Tensor:                        # Smoothed frame at the specified index
        result = data[frame_index].clone()
        seq_len, feature_dim = data.shape
        half_window = self.window_size // 2
        kernel = self.kernel.cpu().numpy()
        
        # For each dimension
        for dim in range(feature_dim):
            feature_seq = data[:, dim].cpu().numpy()
            
            # Define window boundaries
            left = max(0, frame_index - half_window)
            right = min(seq_len, frame_index + half_window + 1)
            window = feature_seq[left:right]
            
            # Pad window if needed (near sequence boundaries)
            if len(window) < self.window_size:
                if frame_index < half_window:  # Near start
                    pad_left = half_window - frame_index
                    window = np.pad(window, (pad_left, 0), mode='edge')
                else:  # Near end
                    pad_right = self.window_size - len(window)
                    window = np.pad(window, (0, pad_right), mode='edge')
            
            # Apply kernel
            smoothed_value = np.sum(window * kernel)
            
            # Blend original and smoothed based on smoothing factor
            original_value = feature_seq[frame_index]
            blended_value = (1 - self.smoothing_factor) * original_value + self.smoothing_factor * smoothed_value
            
            result[dim] = torch.tensor(blended_value, device=data.device, dtype=data.dtype)
        
        return result