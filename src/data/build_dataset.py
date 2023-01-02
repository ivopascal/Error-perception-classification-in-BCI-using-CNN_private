from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.dataset import random_split
import os

from tqdm import tqdm

from settings import VALIDATION_PERCENTAGE, SEED, FEEDBACK_WINDOW_OFFSET
from src.data.util import open_file_pickle
from src.util.dataclasses import EpochedDataSet, TimeSeriesRun
from src.util.util import milliseconds_to_samples


def get_available_pickle_folders(path: str, folder_type="") -> Tuple[List[str], List[str]]:
    available_files = []
    available_files_types = []
    for folder in os.listdir(path):
        if os.path.isdir(path + folder):
            available_files_types.append(
                "   > {}\t\t{}\t{} (directory)\n".format(len(available_files), folder_type, folder))
            available_files.append(path + folder + "/")
        else:
            available_files_types.append("   > {}\t\t{}\t{} (file)\n".format(len(available_files), folder_type, folder))
            available_files.append(path + folder)
    return available_files, available_files_types


def build_dataset(filepath: Optional[str] = None, dataset: Optional[EpochedDataSet] = None):
    if not dataset:
        dataset = open_file_pickle(filepath)

    x = np.array(dataset.data, dtype="float32")
    train_idxs, val_idxs, test_idxs = split_all_subject_by_session(dataset.labels)

    return apply_split_indices(x, dataset.labels, split_indices=(train_idxs, val_idxs, test_idxs))


def build_continuous_dataset(folderpath: str):
    train_val_x = []
    train_val_y = []
    test_x = []
    test_y = []
    for idx, file_name in enumerate(tqdm(os.listdir(folderpath), unit=' session')):

        run: TimeSeriesRun = open_file_pickle(folderpath + "/" + file_name)

        sess_idx = run.file_sub_sess_run[1]

        # Get list of feedback events at pairs [Feedback_type, Time_sample]
        feedback_events = np.array([list(label) for label in zip(run.labels[:, 1], run.feedback_indices)])

        y = np.full(run.session.shape[1], -1)

        feedback_event_indices = feedback_events[:, 1]
        feedback_event_indices += milliseconds_to_samples(FEEDBACK_WINDOW_OFFSET)
        y[feedback_event_indices] = feedback_events[:, 0]
        x_block = np.array(run.session, dtype="float32")
        y_block = [run.file_sub_sess_run[0], sess_idx, 1, run.file_sub_sess_run[2], y]
        if sess_idx == 1:
            train_val_x.append(x_block)
            train_val_y.append(y_block)
        elif sess_idx == 2:
            test_x.append(x_block)
            test_y.append(y_block)
        else:
            raise ValueError("Session index not 1 or 2")

    # Split values train val!
    x_train, x_val, y_train, y_val = train_test_split(train_val_x, train_val_y,
                                                      test_size=VALIDATION_PERCENTAGE,
                                                      random_state=SEED)
    return (x_train, y_train), (x_val, y_val), (test_x, test_y)


def _apply_single_indices(x, y, indices):
    result_set = []
    for idx in indices:
        result_set.append([x[idx], y[idx]])
    return result_set


def apply_split_indices(x, y, split_indices):
    return [_apply_single_indices(x, y, idxs) for idxs in split_indices]


def split_all_subject_by_session(data_labels):
    # One of the sessions will be used for training alone
    # Specify the test and validation relative sizes for that session (sum up to 1)

    # Define train and test sessions
    train_val_session = 1
    test_session = 2

    train_val_idxs = np.where(data_labels[:, 1] == train_val_session)[0]
    test_idxs = np.where(data_labels[:, 1] == test_session)[0]

    # Separate train dataset into train and validation
    train_val_size = len(train_val_idxs)
    val_norm_size = int(train_val_size * VALIDATION_PERCENTAGE)
    train_norm_size = train_val_size - val_norm_size
    train_idxs, val_idxs = random_split(train_val_idxs, [train_norm_size, val_norm_size])

    return train_idxs, val_idxs, test_idxs
