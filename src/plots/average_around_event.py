from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from settings import CONTINUOUS_TESTING_INTERVAL
from src.util.util import milliseconds_to_samples, samples_to_milliseconds


def _collect_recording_around_event(metrics, lower_window, upper_window, y_to_plot):
    y_true_id = 1 - metrics.y_true.clone().double()
    y_true_id[~metrics.y_in_distribution] = 0.5

    event_indices = np.where(metrics.y_in_distribution.numpy())

    y_variances = []
    y_predictions = []
    y_trues = []
    y_subj = []
    for event_index in range(len(event_indices[0])):
        y_true = 1 - (metrics.y_true[event_indices[0][event_index]])

        if y_to_plot is not None and y_to_plot != y_true:
            continue

        y_subj.append(metrics.y_subj_idx[
                      event_indices[0][event_index] + lower_window: event_indices[0][event_index] + upper_window])
        y_variances.append(metrics.y_variance[
                           event_indices[0][event_index] + lower_window: event_indices[0][event_index] + upper_window])
        y_predictions.append(1 - metrics.y_predicted[event_indices[0][event_index] + lower_window:
                                                     event_indices[0][event_index] + upper_window])
        y_trues.append(
            y_true_id[event_indices[0][event_index] + lower_window: event_indices[0][event_index] + upper_window])

    stacked_subjects = torch.vstack(y_subj)
    stacked_variances = torch.vstack(y_variances)
    stacked_predictions = torch.vstack(y_predictions)
    stacked_trues = torch.vstack(y_trues)

    return stacked_subjects, stacked_variances, stacked_predictions, stacked_trues


def _separate_recordings_per_participant(stacked_subjects, stacked_variances, stacked_predictions, stacked_trues):
    predictions_per_participant = []
    variances_per_participant = []
    trues_per_participant = []

    for i in range(1, 7):
        predictions_per_participant.append(
            stacked_predictions[stacked_subjects == i].reshape(-1, stacked_subjects.shape[1]).mean(axis=0))
        variances_per_participant.append(
            stacked_variances[stacked_subjects == i].reshape(-1, stacked_subjects.shape[1]).mean(axis=0))
        trues_per_participant.append(
            stacked_trues[stacked_subjects == i].reshape(-1, stacked_subjects.shape[1]).mean(axis=0))

    return torch.vstack(predictions_per_participant), \
           torch.vstack(variances_per_participant), \
           torch.vstack(trues_per_participant)


def plot_average_around_event(metrics, lower_window_ms, upper_window_ms, testing_interval=CONTINUOUS_TESTING_INTERVAL,
                              y_to_plot=None, label_prefix="", figax=None) -> Tuple[Figure, Axes]:
    lower_window = int(milliseconds_to_samples(lower_window_ms) / testing_interval)
    upper_window = int(milliseconds_to_samples(upper_window_ms) / testing_interval)
    x = [samples_to_milliseconds(step) * testing_interval for step in range(lower_window, upper_window)]

    recordings = _collect_recording_around_event(metrics,
                                                 lower_window,
                                                 upper_window,
                                                 y_to_plot)
    predictions_per_participant, variances_per_participant, trues_per_participant = \
        _separate_recordings_per_participant(*recordings)

    avg_variances = variances_per_participant.mean(axis=0)
    std_variances = variances_per_participant.std(dim=0)

    avg_predictions = predictions_per_participant.mean(axis=0)
    std_predictions = predictions_per_participant.std(dim=0)

    avg_trues = trues_per_participant.mean(axis=0)
    std_trues = trues_per_participant.std(dim=0)

    if not figax:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    ax.plot(x, avg_variances, label=f"{label_prefix}Uncertainty")
    ax.fill_between(x, avg_variances - std_variances, avg_variances + std_variances, label=f"_{label_prefix}Variances",
                    alpha=0.1)

    ax.plot(x, avg_predictions, label=f"{label_prefix}Predictions")
    ax.fill_between(x, avg_predictions - std_predictions, avg_predictions + std_predictions,
                    label=f"_{label_prefix}Predictions", alpha=0.1)

    ax.plot(x, avg_trues, label=f"{label_prefix}Truths")
    ax.fill_between(x, avg_trues - std_trues, avg_trues + std_trues, label=f"_{label_prefix}Truths", alpha=0.1)

    ax.legend()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Variance / prediction")
    ax.set_ylim(0, 1)

    return fig, ax
