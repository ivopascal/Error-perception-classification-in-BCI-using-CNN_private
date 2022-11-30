from dataclasses import dataclass
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
    file_sub_sess_run: Any = None  # This should be a dataclass of participant_idx, session_idx, trial_idx, y_label
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
class EvaluationMetrics:
    y_true: Tensor
    y_predicted: Tensor
    y_variance: Tensor
    y_in_distribution: Tensor
    y_true_matrix: np.array
    y_predicted_matrix: np.array
    statscores: StatScores
    precision: Tensor
    recall: Tensor
    negative_predictive_value: float
    accuracy_conf_matrix: float
    f1_score: Tensor
    mcc: float
    n_mcc: float

