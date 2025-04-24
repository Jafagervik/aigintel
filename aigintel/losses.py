from tinygrad.nn import Tensor


def mse(x: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Mean Squared Error (MSE)"""
    if reduction == "mean":
        return ((x - y) ** 2).mean()
    if reduction == "sum":
        return ((x - y) ** 2).sum()
    if reduction == "none":
        return (x - y) ** 2
    else:
        raise NotImplementedError


def mae(x: Tensor, y: Tensor, reduction: str = "mean") -> Tensor:
    """Mean Absolute Error (MAE)"""
    if reduction == "mean":
        return ((x - y).abs()).mean()
    if reduction == "sum":
        return ((x - y).abs()).sum()
    if reduction == "none":
        return (x - y).abs()
    else:
        raise NotImplementedError


def log_likelihood_loss(y: Tensor, y_hat: Tensor) -> Tensor:
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).sum().realize()


def total_correlation_loss(z: Tensor, q_z: Tensor) -> Tensor:
    log_q_z_product = (q_z.log()).sum(axis=-1)
    log_q_z = q_z.mean(axis=0).log()
    return (log_q_z - log_q_z_product).mean().realize()
