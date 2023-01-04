from typing import List

from src.util.dataclasses import PerParticipant, PredLabels


def split_metrics_per_participant(pred_labels: PredLabels) -> List[PerParticipant]:
    per_participants = []
    for i in range(1, 7):
        per_participants.append(
            PerParticipant(
                predictions=pred_labels.y_predicted[pred_labels.y_subj_idx == i],
                variances=pred_labels.y_variance[pred_labels.y_subj_idx == i],
                trues=pred_labels.y_true[pred_labels.y_subj_idx == i],
                y_in_distribution=pred_labels.y_in_distribution[pred_labels.y_subj_idx == i],
            )
        )

    return per_participants
