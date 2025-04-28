import logging
import os
from abc import abstractmethod
from typing import List

from tinygrad import Tensor, dtypes, nn
from tinygrad.nn.state import load_state_dict, safe_load, safe_save


class BaseModel:
    """Our AI model"""

    @abstractmethod
    def __init__(self, layers: List[int]):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def parameters(self) -> List[Tensor]:
        return nn.state.get_parameters(self)

    def state_dict(self) -> dict[str, Tensor]:
        return nn.state.get_state_dict(self)

    def half(self):
        """Converting all"""
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.half))

    def float(self):
        for param in self.state_dict().values():
            param.replace(param.cast(dtypes.float32))

    @property
    def l1reg(self):
        return sum(p.abs().sum for p in self.parameters())

    @property
    def l2reg(self):
        return sum((p**2).sum() for p in self.parameters())

    def save(self):
        """Saving model to checkpoints"""
        path = os.path.join(os.getcwd(), "checkpoints", f"{self.name.lower()}.safetensors")
        safe_save(self.state_dict(), path)
        logging.debug(f"Saving model {self.name.lower()} to {path}")

    def load(self):
        """Loading model from safetensors file"""
        path = os.path.join(os.getcwd(), "checkpoints", f"{self.name.lower()}.safetensors")
        load_state_dict(self, safe_load(path))
        logging.debug(f"Model loaded from {path}")
