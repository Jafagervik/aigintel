import logging
import time
from argparse import Namespace

from tinygrad import TinyJit, Tensor
from tinygrad.helpers import trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Optimizer

from aigintel.early_stopping import EarlyStopping
from aigintel.imgutils import plot_metrics
from aigintel.models.basemodel import BaseModel
from aigintel.utils import (
    save_model,
    load_model,
    select_optimizer,
)

import numpy as np


@TinyJit
@Tensor.train()
def train_step(
    model: BaseModel, x: Tensor, y: Tensor, optim: Optimizer
) -> Tensor:
    """Single training step a batch of data"""
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    return loss


@TinyJit
@Tensor.test()
def get_test_acc(model: BaseModel, x: Tensor, y: Tensor) -> Tensor:
    return (model(x).argmax(axis=1) == y).mean() * 100


def shuffle_batch(x_train: Tensor, y_train: Tensor) -> (Tensor, Tensor):
    indices = Tensor(
        np.random.permutation(x_train.shape[0])
    )  # Randomly shuffle indices
    return x_train[indices], y_train[indices]


def train(model: BaseModel, config: dict, args: Namespace):
    """Training our latest and greatest AI model"""
    if args.load:
        # Transfer learning
        load_model(model, model.name, args.debug)

    optim = select_optimizer(model, config["optimizer"])

    x_train, y_train, x_test, y_test = mnist(fashion=config["fashion"])

    # Batching parameters
    batch_size = config.get(
        "batch_size", 64
    )  # Default to 64 if not specified in config
    num_batches = (
        x_train.shape[0] // batch_size
    )  # Number of batches (e.g., 60000 // 64 ≈ 937)

    best_loss: float = float("inf")
    best_epoch: int = 0
    test_acc = float("nan")

    es = EarlyStopping(
        config["early_stopping"]["patience"],
        config["early_stopping"]["min_delta"],
    )

    losses = [0 for _ in range(config["epochs"])]
    epoch_durations = [0 for _ in range(config["epochs"])]

    # with Tensor.train():
    for i in (t := trange(config["epochs"])):
        # Shuffle data at the start of each epoch
        x_train, y_train = shuffle_batch(x_train, y_train)

        epoch_loss = 0.0
        start = time.perf_counter()
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size

            x_batch = x_train[start_idx:end_idx].contiguous()
            y_batch = y_train[start_idx:end_idx].contiguous()

            loss = train_step(model, x_batch, y_batch, optim)
            epoch_loss += loss.item()
        duration = time.perf_counter() - start

        avg_loss = epoch_loss / num_batches
        losses[i] = avg_loss
        epoch_durations[i] = duration

        if i % 2 == 0:
            test_acc = get_test_acc(model, x_test, y_test).item()
            t.set_description(
                f"Epoch {i + 1} | Avg Loss = {avg_loss:.4f} | Acc = {test_acc:.2f}%"
            )
        else:
            t.set_description(f"Epoch {i + 1} | Avg Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = i
            save_model(model, "LinearNet")

        if es(avg_loss):
            logging.warn(
                f"Stopping early at epoch {i + 1} since we're not experiencing meaningful decrease of loss"
            )
            break

    logging.info(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    plot_metrics(losses)
    plot_metrics(epoch_durations, "Duration", "Epoch", "Seconds")
