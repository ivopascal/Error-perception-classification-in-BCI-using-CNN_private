from dataclasses import dataclass
from typing import Any, Optional


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
    file_sub_sess_run: Any = None
    labels: Any = None
    feedback_indices: Any = None
    filtered_metadata: Any = None
    file_name: Optional[str] = None
