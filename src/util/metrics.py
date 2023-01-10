import torch
from torch import Tensor


def binary_acc(y_hat, y):
    if y.dim() == 1:
        y_sigmoid = torch.sigmoid(y_hat)
        # define 0.5 as threshold
        y_sigmoid[y_sigmoid >= 0.5] = 1
        y_sigmoid[y_sigmoid < 0.5] = 0

        # Calculate accuracy (sum of all inputs equal to the targets divided by total number of targets)
        correct = (y_sigmoid == y)
    else:
        correct = torch.tensor([y_hat[idx, y[idx]] for idx in range(len(y))])
    correct_results_sum = correct.sum()
    acc = correct_results_sum / len(y)
    acc = torch.round(acc * 100)

    return acc


def beta_nll_loss(mean: Tensor, variance: Tensor, target: Tensor, beta=0.5, epsilon=1e-8):
    """Compute beta-NLL loss
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
    :param epsilon: Small value for numerical stability (avoid log(0) when variance -> 0)

    weighting between data points, where '0' corresponds to
    high weight on low error points and '1' to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    variance += epsilon

    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * variance.detach() ** beta

    if loss.isnan().sum():
        raise ValueError("Computed loss has NaNs")

    return loss.sum(axis=-1)
