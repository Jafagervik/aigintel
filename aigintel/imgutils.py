import matplotlib.pyplot as plt
from tinygrad import Tensor

def show_img(data: Tensor, index: int, show: bool = True, save: bool = False):
    """Open a specific image from a tensor"""
    if index < 0 or index >= data.shape[0]: raise ValueError("Index out of range")

    single_image = data[index]

    imate_np = single_image.numpy().squeeze()
    plt.imshow(imate_np, cmap="gray")
    plt.axis("off")

    if save: plt.savefig(f"images/{index}.png")

    if show: plt.show()

def plot_loss(losses, title="Training Loss Over Epochs", xlabel="Epoch", ylabel="Loss", show: bool = True, save: bool = False):
    """Plot the training loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save: plt.savefig(f"images/{title}.png")
    if show: plt.show()