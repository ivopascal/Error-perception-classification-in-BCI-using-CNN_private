from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.util.dataclasses import EvaluationMetrics


def plot_prediction_variance(metrics: EvaluationMetrics) -> Tuple[Figure, Axes]:
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(metrics.y_predicted[~metrics.y_in_distribution], metrics.y_variance[~metrics.y_in_distribution])
    axs[0].set_title("Out of distribution")
    axs[1].scatter(metrics.y_predicted[metrics.y_in_distribution], metrics.y_variance[metrics.y_in_distribution])
    axs[1].set_title("In distribution")

    axs[0].set_ylabel("Predicted variance")
    for ax in axs:
        ax.set_xlabel("Predicted value")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.55)

    return fig, axs
