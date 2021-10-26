import numpy as np


def calculate_ece(predicted, labels, num_bins=15):
    """
    Calculates the expected calibration error (ECE) for the provided number of bins.

    :param predicted: Probabilistic predictions
    :param labels: Hard labels (classes)
    :param num_bins: Number of bins for the ECE discretization
    :return: Returns the numeric ECE score
    """
    predicted_cls = predicted.argmax(1)

    interval_step = 1. / num_bins
    bin_sizes = np.zeros(num_bins)

    confidences = np.max(predicted, axis=-1)
    bin_indices = np.minimum(confidences // interval_step, num_bins - 1).astype(np.int32)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)

    for i in range(predicted.shape[0]):
        bin_idx = bin_indices[i]
        bin_sizes[bin_idx] += 1

        if predicted_cls[i] == labels[i]:
            bin_accs[bin_idx] += 1
        bin_confs[bin_idx] += confidences[i]

    for i in range(num_bins):
        bin_accs[i] /= max(bin_sizes[i], 1)
        bin_confs[i] /= max(bin_sizes[i], 1)

    ece = 0.
    for i in range(num_bins):
        ece += (bin_sizes[i] / predicted.shape[0]) * np.abs(bin_accs[i] - bin_confs[i])

    return ece
