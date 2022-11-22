from data.build_dataset import save_file_pickle, open_file_pickle
from data.data_wrangling import metadata2path_code
import numpy as np
from tqdm import tqdm

from settings import PROJECT_EPOCHED_FOLDER, FEEDBACK_WINDOW_OFFSET, FEEDBACK_WINDOW_SIZE, SAMPLING_FREQUENCY


def milliseconds_to_samples(milliseconds: int, sampling_frequency: int = SAMPLING_FREQUENCY) -> int:
    return int(milliseconds * sampling_frequency / 1000)


def epoch_data(file_names, file_type):
    # Store trials. Divide into trials that will have always the same size.
    epoched_data = []
    # Store trial labels with format: [#subj, #sess, #trial, label]
    epoched_data_labels = []

    for idx, file_name in enumerate(tqdm(file_names, unit=' session')):

        # Unpack depending on file type:
        if file_type == "Pre-processed":
            (session, file_sub_sess_run, run_labels, fb_indexes, filtered_metadata) = open_file_pickle(file_name)
        elif file_type == "Raw":
            (session, file_sub_sess_run, run_labels, fb_indexes) = open_file_pickle(file_name)
            filtered_metadata = None
        else:
            raise ValueError(f"Unkown file_type {file_type}")

        # Get subject and session numbers
        subj = file_sub_sess_run[0]
        sess = file_sub_sess_run[1]
        run = file_sub_sess_run[2]

        # Get list of feedback events at pairs [Feedback_type, Time_sample]
        fb_events = np.array([list(l) for l in zip(run_labels[:, 1], fb_indexes)])

        # Append individual trial data
        for trial_idx, (label, trial_index) in enumerate(fb_events):
            idx_start = trial_index + milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET)
            idx_end = idx_start + milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
            epoched_data.append(session[:, idx_start:idx_end])
            epoched_data_labels.append([subj, sess, run, (trial_idx + 1), label])

    # After all trials have been added, convert it to Numpy
    epoched_data = np.array(epoched_data)
    epoched_data_labels = np.array(epoched_data_labels)

    epoched_metadata = {
        'fb_windowOnset': FEEDBACK_WINDOW_OFFSET,
        'fb_windowSize': FEEDBACK_WINDOW_SIZE,
        'fb_windowOnsetSamples': milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET),
        'fb_windowSizeSamples': milliseconds_to_samples(FEEDBACK_WINDOW_SIZE)
    }
    # Add metadata

    # File name where to store the pre-processed data
    file_path = PROJECT_EPOCHED_FOLDER + metadata2path_code(filt_metadata=filtered_metadata,
                                                            epoch_metadata=epoched_metadata) + ".p"
    # Save
    save_file_pickle((epoched_data, epoched_data_labels, filtered_metadata, epoched_metadata),
                     file_path)
    print("Saving on " + file_path)

    return file_path
