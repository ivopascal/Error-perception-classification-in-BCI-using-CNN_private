from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.util.dataclasses import PerParticipant


def calculate_best_threshold(fpr, tpr, thresholds) -> Tuple[float, float, int]:
    gmeans = np.sqrt(tpr * (1-fpr))
    best_index = np.argmax(gmeans)

    return thresholds[best_index], gmeans[best_index], best_index


def plot_roc_auc(per_participants: List[PerParticipant]):
    fig, ax = plt.subplots()
    for i, participant in enumerate(per_participants):
        fpr, tpr, thresholds = roc_curve(1 - participant.trues.reshape(-1),
                                         1 - participant.predictions.reshape(-1))
        best_threshold, _, best_index = calculate_best_threshold(fpr, tpr, thresholds)

        ax.plot(fpr, tpr, label=f"P: {i+1}, AUC: {auc(fpr, tpr):.2f}, T: {best_threshold:.2f}")
        ax.plot(fpr[best_index], tpr[best_index], 'b+', mew=2, ms=10)

    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Postivie Rate")
    ax.set_title("ROCs per participant")
    ax.legend()

    return fig, ax


