import jax
import jax.numpy as jnp


__all__ = [
    "evaluate_ece",
]


def evaluate_ece(confidences, true_labels, log_input=True, eps=1e-8, num_bins=15):
    """
    Args:
        confidences (Array): An array with shape [N, K,].
        true_labels (Array): An array with shape [N,].
        log_input (bool): Specifies whether confidences are already given as log values.
        eps (float): Small value to avoid evaluation of log(0) when log_input is False.
        num_bins (int): Specifies the number of bins used by the historgram binning.

    Returns:
        A dictionary of components for expected calibration error.
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    max_confidences = jnp.max(jnp.exp(log_confidences), axis=1)
    max_pred_labels = jnp.argmax(log_confidences, axis=1)
    raw_accuracies = jnp.equal(max_pred_labels, true_labels)
    
    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[ 1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_frequencies = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = jnp.logical_and(max_confidences > bin_lower, max_confidences <= bin_upper)
        bin_frequencies.append(jnp.sum(in_bin))
        if bin_frequencies[-1] > 0:
            bin_accuracies.append(jnp.mean(raw_accuracies[in_bin]))
            bin_confidences.append(jnp.mean(max_confidences[in_bin]))
        else:
            bin_accuracies.append(None)
            bin_confidences.append(None)
    
    bin_accuracies = jnp.array(bin_accuracies)
    bin_confidences = jnp.array(bin_confidences)
    bin_frequencies = jnp.array(bin_frequencies)

    return {
        'bin_lowers': bin_lowers,
        'bin_uppers': bin_uppers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_frequencies': bin_frequencies,
        'ece': jnp.nansum(
            jnp.abs(
                bin_accuracies - bin_confidences
            ) * bin_frequencies / jnp.sum(bin_frequencies)
        ),
    }
