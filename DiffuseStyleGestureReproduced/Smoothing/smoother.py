import torch
from torch import Tensor


# An interface / Abstract class for smoothing operations. 
# This is used outside the inference loop, as a last effort to smooth the movement data.
class Smoother:
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor

    def smooth_sequence(self, data: Tensor):
        """
        Smooth the input data.

        :param data: The input data to be smoothed.
        :return: The smoothed data.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def smooth_frame(self, data: Tensor, frame_index: int):
        """
        Smooth a specific frame in the input data.

        :param data: The input data to be smoothed.
        :param index: The index of the frame to be smoothed.
        :return: The smoothed frame at the specified index.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def set_smoothing_factor(self, factor):
        """
        Set the smoothing factor.

        :param factor: The new smoothing factor.
        """
        self.smoothing_factor = factor

