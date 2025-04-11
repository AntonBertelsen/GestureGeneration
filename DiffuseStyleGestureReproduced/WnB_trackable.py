from abc import ABC, abstractmethod
from typing import Dict, Union

class WnBTrackable(ABC):
    @abstractmethod
    def get_WnB_config_specs(self) -> Dict[str, Union[str, int, float, bool]]:
        # Return the configuration specs needed for Weights & Biases tracking.
        # This should be a dictionary with keys as the parameter names and values as their types.
        # Keys are strings; values can be str, int, float, or bool.
        # Example: return {"learning_rate": float, "batch_size": int}
        pass



