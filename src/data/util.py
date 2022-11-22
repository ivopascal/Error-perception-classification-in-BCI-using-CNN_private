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
