from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.util.dataclasses import EvaluationMetrics, PredLabels


def plot_prediction_variance(pred_labels: PredLabels) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(pred_labels.y_predicted[~pred_labels.y_in_distribution], pred_labels.y_variance[~pred_labels.y_in_distribution])
    axs[0].set_title("Out of distribution")
    axs[1].scatter(pred_labels.y_predicted[pred_labels.y_in_distribution], pred_labels.y_variance[pred_labels.y_in_distribution])
    axs[1].set_title("In distribution")

    axs[0].set_ylabel("Predicted variance")
    for ax in axs:
        ax.set_xlabel("Predicted value")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.55)

    return fig, axs
