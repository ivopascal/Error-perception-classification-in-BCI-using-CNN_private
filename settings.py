import random
import comet_ml

import mne

PROJECT_ROOT_FOLDER = "/Users/ivopascal/Documents/PhD/Error-perception-classification-in-BCI-using-CNN/BCI_root/"
PROJECT_DATASET_FOLDER = PROJECT_ROOT_FOLDER + "Datasets/Monitoring_error-related_potentials_2015/"
PROJECT_DATASET_PICKLE_FOLDER = PROJECT_DATASET_FOLDER + "Datasets_pickle_files/"
PROJECT_BALANCED_FOLDER = PROJECT_DATASET_PICKLE_FOLDER + "Balanced/"
PROJECT_EPOCHED_FOLDER = PROJECT_DATASET_PICKLE_FOLDER + "Epoched/"
PROJECT_IMAGES_FOLDER = PROJECT_ROOT_FOLDER + "Images/"
PROJECT_MODELS_FOLDER = PROJECT_ROOT_FOLDER + "Models/"
PROJECT_RESULTS_FOLDER = PROJECT_ROOT_FOLDER + "Results/"
PROJECT_MODEL_SAVES_FOLDER = PROJECT_MODELS_FOLDER + "Model Saves/"
LOCAL_DATASET_ALL_FOLDER = PROJECT_DATASET_FOLDER + "all/"
PROJECT_RAW_FOLDER = PROJECT_DATASET_PICKLE_FOLDER + "Raw/"
PROJECT_PREPROCESSED_FOLDER = PROJECT_DATASET_PICKLE_FOLDER + "Pre-processed/"

# Properties from data
CHANNEL_NAMES = ["Time", "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5",
                 "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz",
                 "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
                 "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10",
                 "PO8", "PO4", "O2", "Feedback"]
NON_PHYSIOLOGICAL_CHANNELS = {'Time', 'Feedback'}

SUBJECTS_IDX = range(1, 7)
SESSIONS_IDX = [1, 2]
RUNS_IDX = range(1, 11)
SAMPLING_FREQUENCY = 512

# Preprocessing settings
FEEDBACK_WINDOW_OFFSET = 0
FEEDBACK_WINDOW_SIZE = 600  # time in ms

USE_BANDPASS = True
BANDPASS_ORDER = 6
BANDPASS_LOW_FREQ = 1
BANDPASS_HIGH_FREQ = 10
USE_CAUSAL_BUTTERWORTH = True

FILTER_ICA = True
N_ICA_COMPONENTS = 15  # preprocessing 15 already takes about 10 minutes. Same as n_channels is optimal, but expensive
EOG_CHANNEL = "Fpz"  # These should be a difference between two electrodes so you get a derivative
ECG_CHANNEL = "Fpz"
EOG_THRESHOLD = 1.1
HEOG_THRESHOLD = 0.6
ECG_THRESHOLD = 1.1
MUSCLE_THRESHOLD = 1.1
MONTAGE = 'standard_1020'  # 3D layout of channels
mne.set_log_level("ERROR")

EXCLUDE_CHANNELS = None
INCLUDE_CHANNELS = ["FC1", "FCz", "FC2", "C1", "Cz", "C2", "CP3", "CP1", "CPz", "CP2"]
OVERRIDE_SAVES = True
BALANCE_DATASET = True

# Training settings
EXPERIMENT_NAME = "LDA_FCzCz"
DEBUG_MODE = False
OVERRIDEN_HYPER_PARAMS = {
}

MODEL_CLASS_NAME = "disentangled.DisentangledModel"

MODEL_TYPE = "SKLearn"  # Either "Pytorch" or "SKLearn"

# Validation is taken from train sessions (1), so test sessions (2) remain black box
VALIDATION_PERCENTAGE = 0.1
EARLY_STOPPING_PATIENCE = 10
# Evaluate settings
CKPT_PATH = 'last'

SEED = 42  # Not all stochastic processes are seeded yet!
random.seed(SEED)
CONTINUOUS_TEST_BATCH_SIZE = 2048
CONTINUOUS_TESTING_INTERVAL = 10

ENSEMBLE_SIZE = 5

if DEBUG_MODE:
    OVERRIDEN_HYPER_PARAMS["max_num_epochs"] = 5
    EXPERIMENT_NAME = EXPERIMENT_NAME + "_mock"
    if ENSEMBLE_SIZE > 1:
        ENSEMBLE_SIZE = 2
    CONTINUOUS_TESTING_INTERVAL = 30

LOG_DISENTANGLED_UNCERTAINTIES_ON = []  # log ale and epi during train/val but is very slow

print("Settings loaded")
