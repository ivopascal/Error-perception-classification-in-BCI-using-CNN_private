import random
from typing import Optional

from src.data.util import open_file_pickle, save_file_pickle
import numpy as np

from src.data.data_wrangling import metadata2path_code
from settings import PROJECT_BALANCED_FOLDER
from src.util.dataclasses import EpochedDataSet


def oversampling(file_path: Optional[str] = None, epoched_data: Optional[EpochedDataSet] = None) -> EpochedDataSet:
    if not epoched_data:
        epoched_data = open_file_pickle(file_path)

    positive_feedbacks = np.count_nonzero(epoched_data.labels[:, 4] == 1)
    negative_feedbacks = np.count_nonzero(epoched_data.labels[:, 4] == 0)

    diff = positive_feedbacks - negative_feedbacks

    if diff > 0:
        class_idx = 0
        minority_count = negative_feedbacks
    else:
        class_idx = 1
        minority_count = positive_feedbacks

    class_idxs = np.where(epoched_data.labels[:, 4] == class_idx)[0]
    idxs = np.repeat(class_idxs.tolist(), abs(diff) // minority_count)
    remain = abs(diff) % minority_count
    np.random.shuffle(idxs)

    # Select random subset from chosen class
    class_idxs = np.where(epoched_data.labels[:, 4] == class_idx)[0]

    # Add remaining examples
    idxs = np.concatenate([idxs, random.sample(class_idxs.tolist(), remain)])
    data_clones = epoched_data.data[idxs]
    data_labels_clones = epoched_data.labels[idxs]
    # Add
    balanced_data = np.concatenate([epoched_data.data, data_clones])
    balanced_data_labels = np.concatenate([epoched_data.labels, data_labels_clones])

    # 'added_to_class': to which class were the trials added? 0:ErrP; 1:NoErrP
    balanced_metadata = {'clones_added': abs(diff), 'added_to_class': class_idx,
                         'replicate_fold': abs(diff) / minority_count}

    # File name where to store the balanced data
    file_name = metadata2path_code(filt_metadata=epoched_data.filtered_metadata,
                                   epoch_metadata=epoched_data.epoched_metadata,
                                   bal_metadata=balanced_metadata)
    file_path = PROJECT_BALANCED_FOLDER + file_name + ".p"

    balanced_dataset = EpochedDataSet(balanced_data,
                                      balanced_data_labels,
                                      epoched_data.filtered_metadata,
                                      epoched_data.epoched_metadata,
                                      balanced_metadata,
                                      file_name)

    save_file_pickle(balanced_dataset, file_path)

    return balanced_dataset
