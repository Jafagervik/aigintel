from abc import abstractmethod
from typing import List

from tinygrad import Tensor, dtypes, nn


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
