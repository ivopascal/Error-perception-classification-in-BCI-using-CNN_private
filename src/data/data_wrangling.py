import math
import os
from functools import cache
from typing import List, Tuple

import pandas as pd
import numpy as np
import re

from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

from src.data.util import save_file_pickle, open_file_pickle
from settings import PROJECT_DATASET_FOLDER, LOCAL_DATASET_ALL_FOLDER, CHANNEL_NAMES, PROJECT_RAW_FOLDER, SUBJECTS_IDX,\
    SESSIONS_IDX, RUNS_IDX, SAMPLING_FREQUENCY, USE_BANDPASS, BANDPASS_HIGH_FREQ, BANDPASS_LOW_FREQ, BANDPASS_ORDER, \
    NON_PHYSIOLOGICAL_CHANNELS, EXCLUDE_CHANNELS, INCLUDE_CHANNELS, PROJECT_PREPROCESSED_FOLDER
from src.data.util import file_names_timeseries_to_iterator
from src.util.dataclasses import TimeSeriesRun


def indices_to_leading_zeros(indices, n_digits):
    return [str(index).zfill(n_digits) for index in indices]


@cache
def channel_name_to_index(channel_name: str) -> int:
    return CHANNEL_NAMES.index(channel_name)


@cache
def channel_name_to_physiological_index(channel_name: str) -> int:
    physiological_channels = CHANNEL_NAMES
    for non_physiological_channel in NON_PHYSIOLOGICAL_CHANNELS:
        physiological_channels.remove(non_physiological_channel)
    return physiological_channels.index(channel_name)


def load_dataset_from_csv_to_pickle(override_saved_files=True) -> List[TimeSeriesRun]:
    load_labels = pd.read_csv(PROJECT_DATASET_FOLDER + "AllLabels.csv").to_numpy()
    all_labels = []
    for label in load_labels:
        # Get Subject, Session and Feedback number
        trial_metadata = [int(s) for s in re.split('(\d+)', label[0]) if s.isdigit()]
        all_labels.append(np.concatenate([trial_metadata, label[1]], axis=None))
    all_labels = np.array(all_labels)

    # Keep only labels for selected subjects and sessions
    labels = []
    for subj in SUBJECTS_IDX:
        subj_mask = all_labels[:, 0] == subj
        for sess in SESSIONS_IDX:
            sess_mask = all_labels[:, 1] == sess
            idxs = np.where(subj_mask & sess_mask)[0]
            labels.append(all_labels[idxs])
    labels = np.array(np.concatenate(labels))

    file_names = ["Subject" + subject + "_s" + session + "_run" + run
                  for subject in indices_to_leading_zeros(SUBJECTS_IDX, 2)
                  for session in indices_to_leading_zeros(SESSIONS_IDX, 1)
                  for run in indices_to_leading_zeros(RUNS_IDX, 1)]

    datasets = []
    for file_name in tqdm(file_names, unit="file"):
        save_path = PROJECT_RAW_FOLDER + file_name + ".p"
        if os.path.exists(save_path) and not override_saved_files:
            datasets.append(open_file_pickle(save_path))
            continue

        file_path = LOCAL_DATASET_ALL_FOLDER + file_name + ".csv"
        file_sub_sess_run = [int(s) for s in re.split('(\d+)', file_name) if s.isdigit()]
        # Read file and convert to Numpy (transpose to put Channels in the first axis and Time in the second axis)
        new_sess = pd.read_csv(file_path).to_numpy().transpose()
        # Get labels for this session: [trial#, feedback]
        this_indexes = np.where((labels[:, 0] == file_sub_sess_run[0]) & (labels[:, 1] == file_sub_sess_run[1]) & (
                labels[:, 2] == file_sub_sess_run[2]))[0]
        this_labels = labels[this_indexes, 3:]
        # Get feedback indexes
        fb_indexes = np.where(new_sess[channel_name_to_index('Feedback'), :] == 1)[
            0]  # time points where feedback is presented
        dataset = TimeSeriesRun(session=new_sess, file_sub_sess_run=file_sub_sess_run, labels=this_labels,
                                feedback_indices=fb_indexes, file_name=file_name)
        datasets.append(dataset)
        save_file_pickle(dataset, save_path, force_overwrite=True)

    return datasets


def construct_metadata():
    metadata = {}
    if USE_BANDPASS:
        bandpass_metadata = {'low_freq': BANDPASS_LOW_FREQ, 'high_freq': BANDPASS_HIGH_FREQ,
                             'fs': SAMPLING_FREQUENCY,
                             'order': BANDPASS_ORDER}

        metadata.update({"bandpass_filter": bandpass_metadata})

    if EXCLUDE_CHANNELS or INCLUDE_CHANNELS:
        if EXCLUDE_CHANNELS:
            num_channels = len(CHANNEL_NAMES) - len(NON_PHYSIOLOGICAL_CHANNELS) - len(EXCLUDE_CHANNELS)
        else:
            num_channels = len(INCLUDE_CHANNELS)

        channel_selection = {'exclude_channels': EXCLUDE_CHANNELS, 'include_channels': INCLUDE_CHANNELS,
                             'num_channels': num_channels}

        metadata.update({"channel_selection": channel_selection})

    return metadata


def preprocess_data(file_names: List[str] = None, runs: List[TimeSeriesRun] = None, override_save=True)\
        -> Tuple[List[TimeSeriesRun], str]:
    physiological_indices = np.sort([channel_name_to_index(channel) for channel in CHANNEL_NAMES
                                     if channel not in NON_PHYSIOLOGICAL_CHANNELS])

    run_iterator = file_names_timeseries_to_iterator(file_names, runs)

    preprocessed_runs = []
    for idx, run in enumerate(tqdm(run_iterator, unit='run')):
        # Load raw data from Raw folder
        if file_names:
            run = open_file_pickle(PROJECT_RAW_FOLDER + run + ".p")

        run.filtered_metadata = construct_metadata()
        folder_name = PROJECT_PREPROCESSED_FOLDER + metadata2path_code(run.filtered_metadata) + "/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        output_file_path = folder_name + run.file_name + ".p"

        if os.path.exists(output_file_path) and not override_save:
            preprocessed_run = open_file_pickle(output_file_path)
            preprocessed_runs.append(preprocessed_run)
            continue

        # ------------------------ Remove non phisiological signals ------------------------
        run.session = run.session[physiological_indices, :]

        # ------------------------ Bandpass filter ------------------------
        if USE_BANDPASS:
            run.session = butter_bandpass_filter(run.session, BANDPASS_LOW_FREQ, BANDPASS_HIGH_FREQ, SAMPLING_FREQUENCY,
                                                 order=BANDPASS_ORDER,
                                                 axis=1)

        if EXCLUDE_CHANNELS and INCLUDE_CHANNELS:
            raise ValueError("Specify either only include channels or only exclude channels")
        if EXCLUDE_CHANNELS:  # If dictionary is not empty: Exclusion criterion
            channels_indexes = np.sort([channel_name_to_physiological_index(channel) for channel in CHANNEL_NAMES if
                                        channel not in EXCLUDE_CHANNELS])
            run.session = run.session[channels_indexes, :]
        elif INCLUDE_CHANNELS:  # If dictionary is not empty: Inclusion criterion
            channels_indexes = np.sort(
                [channel_name_to_physiological_index(channel) for channel in CHANNEL_NAMES if
                 channel in INCLUDE_CHANNELS])
            run.session = run.session[channels_indexes, :]

        # ------------------------ Save pre-processed data ------------------------

        preprocessed_runs.append(run)
        save_file_pickle(run, output_file_path,
                         force_overwrite=True)
    return preprocessed_runs, os.path.dirname(output_file_path)


def butter_bandpass_filter(data, lowcut, highcut, fs, order, axis=-1):
    # Correct for forward-backward filtering (doubles the order)
    order = math.floor(order / 2)
    # Design
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    # Perform a forward-backward filter: removes phase shift and doubles order of filter
    return sosfiltfilt(sos, data, axis=axis)


def metadata2path_code(filt_metadata=None, epoch_metadata=None, bal_metadata=None):
    # Final string
    ans = ""
    spacing = "_"

    # If not processing was done
    if not filt_metadata:
        # ans += "noPreProcessing{}".format(spacing)
        ans += "raw{}".format(spacing)

    # ++++++++++++++++++++ Filter metadata ++++++++++++++++++++
    # Bandpass filter
    if 'bandpass_filter' in filt_metadata:
        bp = filt_metadata['bandpass_filter']
        ans += "bp[low:{},high:{},ord:{}]{}".format(bp['low_freq'], bp['high_freq'], bp['order'], spacing)

    # Noise reduction
    if 'noise...' in filt_metadata:
        noise = filt_metadata['noise...']
        ans += "Noise[...{}...{}]{}".format(noise['...1'], noise['...2'], spacing)

    # Channel selection
    if 'channel_selection' in filt_metadata:
        channel_selection = filt_metadata['channel_selection']
        exclude_channels = channel_selection['exclude_channels']
        include_channels = channel_selection['include_channels']
        num_channels = channel_selection['num_channels']
        txt = "cs[#:{},".format(num_channels)
        if exclude_channels:  # If this dictionary is not empty
            # 'all\{ch1,ch2,...} meaning the considered channels are all except ch1, ch2, ...
            txt += "all\\{{{}}}]{}".format(','.join(exclude_channels), spacing)
        elif include_channels:
            txt += "{{{}}}]{}".format(','.join(include_channels), spacing)
        ans += txt

    # ++++++++++++++++++++ Epoch metadata ++++++++++++++++++++
    if epoch_metadata:
        ans += "epoch[onset:{},size:{}]{}".format(epoch_metadata['fb_windowOnset'], epoch_metadata['fb_windowSize'],
                                                  spacing)

    # ++++++++++++++++++++ Balanced metadata ++++++++++++++++++++
    if bal_metadata:
        ans += "bal[added_to:{},#:{}]{}".format(bal_metadata['added_to_class'], bal_metadata['clones_added'], spacing)

    # Remove last spacing
    ans = ans[:-len(spacing)]

    return ans
