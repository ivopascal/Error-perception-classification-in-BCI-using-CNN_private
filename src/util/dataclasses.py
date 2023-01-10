from dataclasses import dataclass, fields
from typing import Any, Optional
import numpy as np
from torch import Tensor


@dataclass
class EpochedDataSet:
    data: Any = None
    labels: Any = None
    filtered_metadata: Any = None
    epoched_metadata: Any = None
    balanced_metadata: Any = None
    file_name: Optional[str] = None


@dataclass
class TimeSeriesRun:
    session: Any = None
    file_sub_sess_run: Any = None  # This should be a dataclass of participant_idx, session_idx, trial_idx
    labels: Any = None
    feedback_indices: Any = None
    filtered_metadata: Any = None
    file_name: Optional[str] = None


@dataclass
class StatScores:
    tp: int
    fp: int
    tn: int
    fn: int
    support: int


@dataclass
class PredLabels:
    y_true: Tensor
    y_predicted: Tensor
    y_variance: Optional[Tensor]
    y_epi_uncertainty: Optional[Tensor]
    y_ale_uncertainty: Optional[Tensor]
    y_in_distribution: Tensor
    y_subj_idx: Tensor

    def __post_init__(self):
        for field in fields(PredLabels):
            value = getattr(self, field.name)
            if value is not None:
                setattr(self, field.name,
                        getattr(self, field.name).reshape(-1).to('cpu'))


@dataclass
class EvaluationMetrics:
    pred_labels: PredLabels
    y_true_matrix: np.array
    y_predicted_matrix: np.array
    statscores: StatScores
    precision: Tensor
    specificity: Tensor
    accuracy: Tensor
    recall: Tensor
    negative_predictive_value: float
    accuracy_conf_matrix: float
    f1_score: Tensor
    mcc: float
    n_mcc: float


@dataclass
class PerParticipant:
    predictions: Tensor
    variances: Tensor
    trues: Tensor
    y_in_distribution: Tensor
