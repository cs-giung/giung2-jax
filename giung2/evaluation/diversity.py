import jax
import jax.numpy as jnp


__all__ = [
    "compute_pairwise_kld",
    "compute_pairwise_agr",
]


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = (labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,)))
    x = jax.lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def compute_pairwise_kld(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, M, K,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of pairwise KL divergence (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise KL divergence values
        (per example) with shape [N, M, M,] when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    n_datapoint = log_confidences.shape[0]
    n_ensembles = log_confidences.shape[1]
    raw_results = jnp.array([
        jnp.sum(
            jnp.multiply(
                jnp.exp(log_confidences[:, idx, :]),
                log_confidences[:, idx, :] - log_confidences[:, jdx, :],
            ), axis=1,
        ) for idx in range(n_ensembles) for jdx in range(n_ensembles)
    ]).reshape(n_ensembles, n_ensembles, n_datapoint).transpose(2, 0, 1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(jnp.zeros(1)) if n_ensembles == 1 else jnp.sum(
            jnp.mean(raw_results, axis=0)
        ) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def compute_pairwise_agr(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences (Array): An array with shape [N, M, K,].
        log_input (bool, unused): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of pairwise agreement (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise agreement values
        (per example) with shape [N, M, M,] when reduction in ["none",].
    """
    pred_labels = jnp.argmax(confidences, axis=2) # [N, M,]
    pred_labels = onehot(pred_labels, confidences.shape[2]) # [N, M, K,]
    n_datapoint = pred_labels.shape[0]
    n_ensembles = pred_labels.shape[1]
    raw_results = jnp.array([
        jnp.sum(
            jnp.multiply(
                pred_labels[:, idx, :],
                pred_labels[:, jdx, :],
            ), axis=1,
        ) for idx in range(n_ensembles) for jdx in range(n_ensembles)
    ]).reshape(n_ensembles, n_ensembles, n_datapoint).transpose(2, 0, 1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(jnp.ones(1)) if n_ensembles == 1 else (
            jnp.sum(jnp.mean(raw_results, axis=0)) - n_ensembles
        ) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
