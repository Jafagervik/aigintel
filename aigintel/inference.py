import logging
from argparse import Namespace
from typing import Dict

from tinygrad.nn.datasets import mnist

from aigintel.imgutils import show_img
from aigintel.models.basemodel import BaseModel
from aigintel.utils import load_fashion_mnist_class_names


def run(model: BaseModel, config: Dict, args: Namespace, show: bool = False):
    """Runs a single forward pass through loaded model"""
    model.load()

    fash = load_fashion_mnist_class_names()

    idx = 2

    _, _, xt, yt = mnist(fashion=True)

    first = xt[idx].unsqueeze(0)

    y = model(first)

    y_idx = y.argmax(axis=1).item()

    logging.info(f"Predicted: {fash[y_idx]}, Actual: {fash[yt[idx].item()]}")

    show_img(first, True, False)


def run_multiple(model: BaseModel, config: Dict, args: Namespace, show: bool = False):
    """Runs a multiple forward passes through loaded model"""
    model.load()

    fash = load_fashion_mnist_class_names()

    _, _, xt, yt = mnist(fashion=True)

    for i in range(1, 10):
        first = xt[i].unsqueeze(0)

        y = model(first)

        # print(yt[0].item())

        y_idx = y.argmax(axis=1).item()

        # print(y_idx)

        logging.info(f"Predicted: {fash[y_idx]}, Actual: {fash[yt[i].item()]}")

        # show_img(first, True, False)
