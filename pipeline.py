from preprocess import preprocess
from train_test import train


def main():
    dataset, continous_dataset_path = preprocess()
    train(dataset=dataset, continous_dataset_path=continous_dataset_path)


if __name__ == "__main__":
    main()