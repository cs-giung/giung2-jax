import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


__all__ = [
    "temperature_scaling",
    "get_optimal_temperature",
    "evaluate_acc",
    "evaluate_nll",
    "compute_ent",
]


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def temperature_scaling(confidences, temperature, log_input=True, eps=1e-8):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        temperature (float): Specifies temperature value for smoothing.
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.

    Returns:
        An array of temperature-scaled confidences or log-confidences with shape [N, K,].
    """
    if log_input:
        # it returns temperature-scaled log_confidences when log_input is True.
        return jax.nn.log_softmax(confidences / temperature, axis=-1)
    else:
        # it returns temperature-scaled confidences when log_input is False.
        return jax.nn.softmax(jnp.log(confidences + eps) / temperature, axis=-1)


def get_optimal_temperature(confidences, true_labels, log_input=True, eps=1e-8):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool, unused): Specifies whether confidences are already given as log values.
        eps (float, unused): Small value to avoid evaluation of log(0) when log_input is False.

    Returns:
        An array of temprature with shape [1,] which minimizes negative log-likelihood for given
        temperature-scaled confidences and true_labels.
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    def obj(t):
        return evaluate_nll(
            temperature_scaling(
                log_confidences, t, log_input=True
            ), true_labels, log_input=True
        )
    optimal_temperature = minimize(obj, jnp.asarray([1.0,]), method='BFGS', tol=1e-3).x[0]
    return optimal_temperature


def evaluate_acc(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool, unused): Specifies whether confidences are already given as log values.
        eps (float, unused): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of accuracy with shape [1,] when reduction in ["mean", "sum",], or raw accuracy
        values with shape [N,] when reduction in ["none",].
    """
    pred_labels = jnp.argmax(confidences, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def evaluate_nll(confidences, true_labels, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of negative log-likelihood with shape [1,] when reduction in ["mean", "sum",], or
        raw negative log-likelihood values with shape [N,] when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    true_target = onehot(true_labels, num_classes=log_confidences.shape[1])
    raw_results = -jnp.sum(true_target * log_confidences, axis=-1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def compute_ent(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of entropy with shape [1,] when reduction in ["mean", "sum",], or
        raw entropy values with shape [N,] when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    raw_results = -jnp.sum(jnp.exp(log_confidences) * log_confidences, axis=-1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
