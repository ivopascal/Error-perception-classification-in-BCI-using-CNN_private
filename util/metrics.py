import torch


def binary_acc(y_hat, y):
    correct = torch.tensor([y_hat[idx, y[idx]] for idx in range(len(y))])

    correct_results_sum = correct.sum()
    acc = correct_results_sum / len(y)
    acc = torch.round(acc * 100)

    return acc