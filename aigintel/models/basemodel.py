from abc import abstractmethod
from typing import List

from tinygrad import Tensor, nn


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

    def properties(self) -> List[Tensor]:
        return nn.state.get_parameters(self)

    def state_dict(self) -> dict[str, Tensor]:
        return nn.state.get_state_dict(self)
