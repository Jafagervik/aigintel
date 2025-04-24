from typing import List, Callable

from tinygrad import Tensor, nn
from aigintel.models.basemodel import BaseModel


# ==============================
# TODO: IMPLEMENT HERE
# ==============================
class LinearNet(BaseModel):
    """Our AI model"""

    def __init__(self):
        self.layers: List[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5),
            Tensor.relu,
            nn.Conv2d(32, 32, 5),
            Tensor.relu,
            nn.BatchNorm(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            nn.Conv2d(64, 64, 3),
            Tensor.relu,
            nn.BatchNorm(64),
            Tensor.max_pool2d,
            lambda x: x.flatten(1),
            nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
