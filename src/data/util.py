import os
import pickle as pk
from typing import Optional, List, Union

from src.util.dataclasses import TimeSeriesRun


def file_names_timeseries_to_iterator(file_names: Optional[List[str]], runs: Optional[List[TimeSeriesRun]])\
        -> List[Union[str, TimeSeriesRun]]:
    if file_names:
        run_iterator = file_names
    elif runs:
        run_iterator = runs
    else:
        raise ValueError("Either file_names or runs should be given")

    return run_iterator


def save_file_pickle(data, path, force_overwrite=False):
    if os.path.exists(path) and not force_overwrite:
        return False
    with open(path, "wb") as f:
        pk.dump(data, f)
    return True


def open_file_pickle(path):
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist!")

    with open(path, "rb") as f:
        return pk.load(f)
