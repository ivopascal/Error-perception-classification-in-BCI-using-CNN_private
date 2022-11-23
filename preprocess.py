from settings import OVERRIDE_SAVES, BALANCE_DATASET
from src.data.balancing import oversampling
from src.data.data_wrangling import load_dataset_from_csv_to_pickle, preprocess_data
from src.data.epoching import epoch_data


def preprocess():
    print("Loading dataset from CSVs...")
    raw_runs = load_dataset_from_csv_to_pickle(override_saved_files=OVERRIDE_SAVES)  # This takes about 1 minute

    print("Preprocessing raw data...")
    preprocessed_runs = preprocess_data(runs=raw_runs, override_save=OVERRIDE_SAVES)  # This takes about 30 seconds

    print("Epoching preprocessed data...")
    epoched_data = epoch_data(runs=preprocessed_runs, override_save=OVERRIDE_SAVES)

    if BALANCE_DATASET:
        print("Balancing epoched data...")
        return oversampling(epoched_data=epoched_data)

    return epoched_data


if __name__ == "__main__":
    preprocess()
