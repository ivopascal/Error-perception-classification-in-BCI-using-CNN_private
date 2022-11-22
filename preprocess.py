from src.data.balancing import oversampling
from src.data.data_wrangling import load_dataset_from_csv_to_pickle, preprocess_data
from src.data.epoching import epoch_data


def preprocess():
    override_saves = False
    print("Loading dataset from CSVs...")
    raw_runs = load_dataset_from_csv_to_pickle(override_saved_files=override_saves)  # This takes about 1 minute

    print("Preprocessing raw data...")
    preprocessed_runs = preprocess_data(runs=raw_runs, override_save=override_saves)  # This takes about 30 seconds

    print("Epoching preprocessed data...")
    epoched_data = epoch_data(runs=preprocessed_runs, override_save=override_saves)

    print("Balancing epoched data...")
    balanced_data = oversampling(epoched_data=epoched_data)

    return balanced_data


if __name__ == "__main__":
    preprocess()
