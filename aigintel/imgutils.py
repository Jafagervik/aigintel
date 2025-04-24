import os
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

def plot_metrics(values, title="Loss", xlabel="Epoch", ylabel="Value", save=True, show=True,
                 color='#4361EE', marker_color='#3F37C9', figsize=(10, 6)):
    """Create a modern-looking visualization of training metrics"""
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=figsize, facecolor='#f9f9f9')

    # Plot the main line with a gradient fill
    epochs = range(1, len(values) + 1)
    ax.plot(epochs, values, color=color, lw=2.5, zorder=5)
    ax.fill_between(epochs, values, alpha=0.2, color=color, zorder=4)

    # Add markers at each data point
    ax.scatter(epochs, values, color=marker_color, s=60, zorder=6, alpha=0.8,
               edgecolor='white', linewidth=1.5)

    # Customize the appearance
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    # Customize grid and spines
    ax.grid(True, linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add a subtle text annotation with the min/max values
    min_val = min(values)
    max_val = max(values)
    min_idx = values.index(min_val) + 1
    max_idx = values.index(max_val) + 1

    ax.annotate(f'Min: {min_val:.4f} (Epoch {min_idx})',
                xy=(min_idx, min_val), xytext=(10, -30),
                textcoords='offset points', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    # Only annotate max if it's different from min (for accuracy it might be the most important)
    if max_val != min_val:
        ax.annotate(f'Max: {max_val:.4f} (Epoch {max_idx})',
                    xy=(max_idx, max_val), xytext=(10, 30),
                    textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    plt.tight_layout()

    if save:
        os.makedirs("images", exist_ok=True)  # Create images directory if it doesn't exist
        plt.savefig(f"images/{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()