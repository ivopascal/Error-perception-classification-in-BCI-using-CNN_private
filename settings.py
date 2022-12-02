import random

from src.Models.ConvNet import ConvNet64C
from src.Models.EegNet import EEGNet, BayesianEEGNet
from src.Models.CorreiaNet import CorreiaNet, BayesianCorreiaNet

# Project folders settings
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

EXCLUDE_CHANNELS = None
INCLUDE_CHANNELS = None
OVERRIDE_SAVES = False
BALANCE_DATASET = True

# Training settings
EXPERIMENT_NAME = "Bayesian_CorreiaNet_Variance"
DEBUG_MODE = True
OVERRIDEN_HYPER_PARAMS = {
    "max_num_epochs": 200,
    "bayesian_forward_passes": 50
}

if DEBUG_MODE:
    OVERRIDEN_HYPER_PARAMS["max_num_epochs"] = 100
    EXPERIMENT_NAME = EXPERIMENT_NAME + "_mock"
MODEL_CLASS = BayesianCorreiaNet

# Validation is taken from train sessions (1), so test sessions (2) remain black box
VALIDATION_PERCENTAGE = 0.1

# Evaluate settings
CKPT_PATH = 'last'

SEED = 42  # Not all stochastic processes are seeded yet!
random.seed(SEED)
CONTINUOUS_TEST_BATCH_SIZE = 2048
CONTINUOUS_TESTING_INTERVAL = 10
