from utils.WnB_trackable import WnBTrackable
import torch
from typing import Callable
import math
from abc import abstractmethod, ABC

class Diffusion(WnBTrackable, ABC):

    @abstractmethod
    def forward(self, sequence_tensor: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_sequence_timesteps(self, current_timestep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")


    @property
    @abstractmethod
    def number_of_timesteps(self) -> int:
        raise NotImplementedError("Subclasses must implement this property.")

    @property
    @abstractmethod
    def clean_frame_index(self) -> int:
        raise NotImplementedError("Subclasses must implement this property.")
    
    @abstractmethod
    def get_WnB_config_specs(self):
        raise NotImplementedError("Subclasses must implement this method.")

    
    ###########################################################################################################################
    
    # noise schedules

    # There are several different noise schedules that can be used.
    # which one to use is a hyperparameter that can be tuned.
    # We detach them from the forward diffusion function, so it is easier to switch between them.

    @staticmethod
    def linear_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # This is a simple linear schedule. Noice is added as a uniform increase.
        # Pros: Simple and works well.
        # Cons: Can be suboptimal, and to0 simple

        def linear_schedule(t: int, T: int) -> float:
            return beta_min + (beta_max - beta_min) * (t / T)
        
        return (linear_schedule, {"name": "linear_schedule", "beta_min": beta_min, "beta_max": beta_max})

    @staticmethod
    def quadratic_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # This makes the diffusion more 'agressive' (high var) when t is high. Starts slow, then increases rapidly.
        # Pro: Preserves details early on
        # Con: High noise at later timesteps

        def quadratic_schedule(t: int, T: int) -> float:
            return beta_min + (beta_max - beta_min) * (t / T) ** 2
            # return (t / T) ** 2

        return (quadratic_schedule, {"name": "quadratic_schedule", "beta_min": beta_min, "beta_max": beta_max})

    @staticmethod
    def exponential_schedule(beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # 'beta' is the hyperparams of the exponentaial schedual.
        # Typical beta values are 0.0001 and 0.02, and they are used to control the rate of exponental decay of the variance.

        def exponential_schedule(t: int, T: int) -> float:
            return beta_min * ((beta_max / beta_min) ** (t / T))

        return (exponential_schedule, {"name": "exponential_schedule", "beta_min": beta_min, "beta_max": beta_max})

    @staticmethod
    def sigmoid_schedule(k = 10, beta_min = 0.0001, beta_max = 0.02) -> Callable[[int, int], float]:
        # 'k' is the hyperparam of the sigmoid schedule.
        # It controls the steepness of the sigmoid function, than inturn controls how fast the variance changes 
        # at the start, end and middel of the diffusion.

        def sigmoid_schedule(t: int, T: int):
            return beta_min + (beta_max - beta_min) / (1 + math.exp(-k * (t / T - 0.5)))
            # return 1 / (1 + math.exp(-k * (t / T - 0.5)))

        return (sigmoid_schedule, {"name": "sigmoid_schedule", "k": k, "beta_min": beta_min, "beta_max": beta_max})