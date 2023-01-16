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


def classification_negative_log_likelihood(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
        Negative log-likelihood or negative log-probability loss/metric.
        Reference: Evaluating Predictive Uncertainty Challenge, Quiñonero-Candela et al, 2006.
        It sums over classes: log(y_pred) for true class and log(1.0 - pred) for not true class, and then takes average across samples.
    """
    y_pred = torch.clamp(y_pred, 1e-6, 1.0 - 1e-6)  # Fairly large eps bc 32 bit precision

    losses = (y_true * y_pred.log() + (1.0 - y_true) * (1.0 - y_pred).log()).sum(dim=-1)
    if losses.mean().isinf():
        print(losses)
    return -losses.mean(dim=-1)