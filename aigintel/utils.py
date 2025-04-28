import argparse
import logging
import os
import random as rnd
from typing import List

import yaml
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Optimizer
from tinygrad.nn.state import load_state_dict, safe_load

from aigintel.models.basemodel import BaseModel


def load_config(path: str) -> dict:
    """Loads hyperparameters from a YAML file."""
    logging.info("Loading hyperparameters")
    with open(path) as f:
        return yaml.safe_load(f)


def select_optimizer(model: BaseModel, config: dict) -> Optimizer:
    """Selects optimizer based on hyperparameters."""
    match config["name"].lower():
        case "sgd":
            return nn.optim.SGD(nn.state.get_parameters(model), lr=config["lr"])
        case "adam":
            return nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])
        case "adamw":
            return nn.optim.AdamW(nn.state.get_parameters(model), lr=config["lr"])
        case _:
            logging.error("Optimizer is not supported.")
            raise NotImplementedError


def load_model(model: BaseModel, model_name: str):
    """Loading model from safetensors file"""
    path = os.path.join(os.getcwd(), "checkpoints", f"{model_name.lower()}.safetensors")
    load_state_dict(model, safe_load(path))
    logging.debug(f"Model loaded from {path}")


def seed_all(seed: int = 1337):
    Tensor.manual_seed(seed)
    rnd.seed(seed)
    logging.info(f"Seed set to {seed}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gintel AI LAB")

    parser.add_argument(
        "--load",
        "-l",
        dest="load",
        action="store_true",
        help="Load a model",
        default=False,
    )

    parser.add_argument(
        "--debug",
        "-d",
        dest="debug",
        action="store_true",
        help="Debug",
        default=False,
    )

    parser.add_argument(
        "--train",
        "-t",
        dest="train",
        action="store_true",
        help="Train mode or inference",
        default=False,
    )

    return parser.parse_args()


def load_fashion_mnist_class_names() -> List[str]:
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]
