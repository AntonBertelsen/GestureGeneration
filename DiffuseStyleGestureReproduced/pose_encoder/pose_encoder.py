import torch
import torch.nn as nn

from utils.WnB_trackable import WnBTrackable
from abc import ABC, abstractmethod

class PoseEncoder(nn.Module, WnBTrackable, ABC):
    
    @abstractmethod
    def encode(self, x, return_logvar=False):
        pass

    @abstractmethod
    def decode(self, z):
        pass