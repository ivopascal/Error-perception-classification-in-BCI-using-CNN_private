from typing import List

from src.util.dataclasses import PerParticipant, EvaluationMetrics


def split_metrics_per_participant(metrics: EvaluationMetrics) -> List[PerParticipant]:
    per_participants = []
    for i in range(1, 7):
        per_participants.append(
            PerParticipant(
                predictions=metrics.y_predicted[metrics.y_subj_idx == i],
                variances=metrics.y_variance[metrics.y_subj_idx == i],
                trues=metrics.y_true[metrics.y_subj_idx == i],
                y_in_distribution=metrics.y_in_distribution[metrics.y_subj_idx == i],
            )
        )

    return per_participants
