import jax
import jax.numpy as jnp


__all__ = [
    "compute_pairwise_kld",
    "compute_pairwise_agr",
    "compute_pairwise_cka",
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


def compute_pairwise_cka(output_vecs, reduction="mean"):
    """
    Args:
        output_vecs (Array): An array with shape [N, M, K,].
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        An array of pairwise centered kernel alignment (averaged over off-diagonal elements) with
        shape [1,] when reduction in ["mean",], or raw pairwise centered kernel alignment values
        with shape [M, M,] when reduction in ["none",].
    """
    n_datapoint = output_vecs.shape[0]
    n_ensembles = output_vecs.shape[1]

    raw_results = []
    for idx in range(n_ensembles):
        for jdx in range(n_ensembles):
            identity_mat = jnp.diag(jnp.ones(n_datapoint))
            centering_mat = identity_mat - jnp.ones((n_datapoint, n_datapoint)) / n_datapoint
            x = output_vecs[:, idx, :]
            y = output_vecs[:, jdx, :]
            cov_xy = jnp.trace(
                x @ jnp.transpose(x) @ centering_mat @ y @ jnp.transpose(y) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            cov_xx = jnp.trace(
                x @ jnp.transpose(x) @ centering_mat @ x @ jnp.transpose(x) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            cov_yy = jnp.trace(
                y @ jnp.transpose(y) @ centering_mat @ y @ jnp.transpose(y) @ centering_mat
            )/ jnp.power(n_datapoint - 1, 2)
            raw_results.append(cov_xy / jnp.sqrt(cov_xx * cov_yy))
    raw_results = jnp.array(raw_results).reshape(n_ensembles, n_ensembles)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.sum(raw_results) / (n_ensembles**2 - n_ensembles)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
