from argparse import Namespace
from typing import Dict

from tinygrad import Tensor

from aigintel.models.basemodel import BaseModel
from aigintel.utils import load_model

def run(model: BaseModel, data: Tensor, config: Dict, args: Namespace):
    """Runs a single forward pass through loaded model"""
    load_model(model, model.name, args.debug)

    y = model(data)

    print(y.shape)

    print("Timing")
