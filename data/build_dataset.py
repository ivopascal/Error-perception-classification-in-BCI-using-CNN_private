from typing import Tuple, List, Optional

import numpy as np
from torch.utils.data.dataset import random_split
import os
import pickle as pk

from settings import PROJECT_BALANCED_FOLDER, PROJECT_EPOCHED_FOLDER


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


def build_dataset(variant: str = "Balanced", filepath: Optional[str] = None):
    if variant == "Balanced":
        if filepath:
            data, data_labels, filtered_metadata, epoched_metadata, balanced_metadata = open_file_pickle(filepath)
        else:
            available_files, _ = get_available_pickle_folders(PROJECT_BALANCED_FOLDER, "Balanced")
            data, data_labels, filtered_metadata, epoched_metadata, balanced_metadata = open_file_pickle(
                available_files[0])

    elif variant == "Epoched":
        if filepath:
            data, data_labels, filtered_metadata, epoched_metadata = open_file_pickle(filepath)
        else:
            available_files, _ = get_available_pickle_folders(PROJECT_EPOCHED_FOLDER, "Epoched")
            data, data_labels, filtered_metadata, epoched_metadata = open_file_pickle(available_files[0])

    else:
        raise ValueError(f"Variant {variant} not known")

    data = np.array(data, dtype="float32")
    train_idxs, val_idxs, test_idxs = split_all_subject_by_session(data_labels)

    return apply_split_indices(data, data_labels, split_indices=(train_idxs, val_idxs, test_idxs))


def apply_split_indices(data, data_labels, split_indices):
    train_idxs, val_idxs, test_idxs = split_indices
    train_set = []
    for train_idx in train_idxs:
        train_set.append([data[train_idx], data_labels[train_idx]])
    val_set = []
    for val_idx in val_idxs:
        val_set.append([data[val_idx], data_labels[val_idx]])
    test_set = []
    for test_idx in test_idxs:
        test_set.append([data[test_idx], data_labels[test_idx]])

    return train_set, val_set, test_set


def split_all_subject_by_session(data_labels):
    # One of the sessions will be used for training alone
    # Specify the test and validation relative sizes for that session (sum up to 1)
    test_perc = 0.5
    val_perc = 0.5

    # Define train and test sessions
    train_session = 1
    test_session = 2

    train_idxs = np.where(data_labels[:, 1] == train_session)[0]
    test_idxs = np.where(data_labels[:, 1] == test_session)[0]

    # Separate train dataset into train and validation
    test_size = len(test_idxs)
    test_norm_size = int(test_size * test_perc)
    val_norm_size = test_size - test_norm_size
    test_idxs, val_idxs = random_split(test_idxs, [test_norm_size, val_norm_size])

    return train_idxs, val_idxs, test_idxs
