import random
from data.build_dataset import open_file_pickle, save_file_pickle
import numpy as np

from data.data_wrangling import metadata2path_code
from settings import PROJECT_BALANCED_FOLDER


def oversampling(file_path: str):

    (epoched_data, epoched_data_labels, filtered_metadata, epoched_metadata) = open_file_pickle(file_path)

    positive_feedbacks = np.count_nonzero(epoched_data_labels[:, 4] == 1)
    negative_feedbacks = np.count_nonzero(epoched_data_labels[:, 4] == 0)

    diff = positive_feedbacks - negative_feedbacks

    if diff > 0:
        class_idx = 0
        original_size = negative_feedbacks
        if negative_feedbacks < abs(diff):  # This seems odd. I don't think we should have this
            class_idxs = np.where(epoched_data_labels[:, 4] == class_idx)[0]
            idxs = np.repeat(class_idxs.tolist(), abs(diff) // negative_feedbacks)
            remain = abs(diff) % negative_feedbacks
            np.random.shuffle(idxs)
    else:
        class_idx = 1
        original_size = positive_feedbacks
        if positive_feedbacks < abs(diff):
            class_idxs = np.where(epoched_data_labels[:, 4] == class_idx)[0]
            idxs = np.repeat(class_idxs.tolist(), abs(diff) // positive_feedbacks)
            remain = abs(diff) % positive_feedbacks
            np.random.shuffle(idxs)

        # Select random subset from chosen class
    class_idxs = np.where(epoched_data_labels[:, 4] == class_idx)[0]

    # Add remaining examples
    idxs = np.concatenate([idxs, random.sample(class_idxs.tolist(), remain)])
    data_clones = epoched_data[idxs]
    data_labels_clones = epoched_data_labels[idxs]
    # Add
    balanced_data = np.concatenate([epoched_data, data_clones])
    balanced_data_labels = np.concatenate([epoched_data_labels, data_labels_clones])

    # ++++++++++++++++++++++++ Initialize variables ++++++++++++++++++++++++
    # Store metadata related to the balancing of the data
    # 'clones_added':   how many trials were added to the class?
    # 'added_to_class': to which class were the trials added? 0:ErrP; 1:NoErrP
    # 'replicate_fold': how many times was the original class size increased by?
    balanced_metadata = {'clones_added': abs(diff), 'added_to_class': class_idx, 'replicate_fold': abs(diff) / original_size}

    # File name where to store the balanced data
    file_name = metadata2path_code(filt_metadata=filtered_metadata, epoch_metadata=epoched_metadata,
                                   bal_metadata=balanced_metadata) + ".p"
    file_name = PROJECT_BALANCED_FOLDER + file_name
    # Save
    save_file_pickle((balanced_data, balanced_data_labels, filtered_metadata, epoched_metadata, balanced_metadata),
                     file_name)

    return file_name
