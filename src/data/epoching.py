import os.path
from typing import List, Optional

from src.data.util import open_file_pickle
from src.data.data_wrangling import metadata2path_code
import numpy as np
from tqdm import tqdm

from settings import PROJECT_EPOCHED_FOLDER, FEEDBACK_WINDOW_OFFSET, FEEDBACK_WINDOW_SIZE, SAMPLING_FREQUENCY, \
    SLIDING_AUGMENTATION_RANGE
from src.data.util import file_names_timeseries_to_iterator, save_file_pickle
from src.util.dataclasses import TimeSeriesRun, EpochedDataSet
from src.util.util import milliseconds_to_samples


def epoch_data(file_names: Optional[List[str]] = None, runs: Optional[List[TimeSeriesRun]] = None,
               override_save=True) -> EpochedDataSet:
    # Store trials. Divide into trials that will have always the same size.
    epoched_data = []
    # Store trial labels with format: [#subj, #sess, #trial, label]
    epoched_data_labels = []

    run_iterator = file_names_timeseries_to_iterator(file_names, runs)

    epoched_metadata = {
        'fb_windowOnset': FEEDBACK_WINDOW_OFFSET,
        'fb_windowSize': FEEDBACK_WINDOW_SIZE,
        'fb_windowOnsetSamples': milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET),
        'fb_windowSizeSamples': milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
    }

    output_file_name = metadata2path_code(filt_metadata=runs[0].filtered_metadata,
                                          epoch_metadata=epoched_metadata)
    output_file_path = PROJECT_EPOCHED_FOLDER + output_file_name + ".p"
    if os.path.exists(output_file_path) and not override_save:
        return open_file_pickle(output_file_path)

    for idx, run in enumerate(tqdm(run_iterator, unit=' session')):

        # Unpack depending on file type:
        if file_names:
            run: TimeSeriesRun = open_file_pickle(run)

        # Get subject and session numbers
        subj_idx = run.file_sub_sess_run[0]
        sess_idx = run.file_sub_sess_run[1]
        run_idx = run.file_sub_sess_run[2]

        # Get list of feedback events at pairs [Feedback_type, Time_sample]
        feedback_events = np.array([list(label) for label in zip(run.labels[:, 1], run.feedback_indices)])

        # Append individual trial data
        for trial_idx, (label, trial_index) in enumerate(feedback_events):
            if sess_idx == 1 and SLIDING_AUGMENTATION_RANGE:
                for augment_offset in range(milliseconds_to_samples(SLIDING_AUGMENTATION_RANGE[0]),
                                            milliseconds_to_samples(SLIDING_AUGMENTATION_RANGE[1]), 10):
                    idx_start = trial_index + milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET) + augment_offset
                    idx_end = idx_start + milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
                    epoched_data.append(run.session[:, idx_start:idx_end])
                    epoched_data_labels.append([subj_idx, sess_idx, run_idx, (trial_idx + 1), label])
            else:
                idx_start = trial_index + milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET)
                idx_end = idx_start + milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
                epoched_data.append(run.session[:, idx_start:idx_end])
                epoched_data_labels.append([subj_idx, sess_idx, run_idx, (trial_idx + 1), label])

    # After all trials have been added, convert it to Numpy
    epoched_data = np.array(epoched_data)
    epoched_data_labels = np.array(epoched_data_labels)

    epoched_dataset = EpochedDataSet(epoched_data, epoched_data_labels, runs[0].filtered_metadata, epoched_metadata,
                                     output_file_name)
    save_file_pickle(epoched_dataset, output_file_path)

    return epoched_dataset
