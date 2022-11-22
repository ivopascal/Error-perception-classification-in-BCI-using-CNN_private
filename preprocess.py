from data.balancing import oversampling
from data.data_wrangling import load_dataset_from_csv_to_pickle, preprocess_data
from data.epoching import epoch_data


def preprocess():
    print("Loading dataset from CSVs...")
    file_names = load_dataset_from_csv_to_pickle()  # This takes about 1 minute

    print("Preprocessing raw data...")
    output_file_paths = preprocess_data(file_names)  # This takes about 30 seconds

    print("Epoching preprocessed data...")
    file_name = epoch_data(output_file_paths, file_type="Pre-processed")

    print("Balancing epoched data...")
    file_name = oversampling(file_name)

    return file_name


if __name__ == "__main__":
    preprocess()
