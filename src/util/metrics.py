import torch


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


