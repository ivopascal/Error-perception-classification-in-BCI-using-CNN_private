from preprocess import preprocess
from train_test import train
from train_test_lda import train_lda


def main():
    dataset, continous_dataset_path = preprocess()
    # train(dataset=dataset, continous_dataset_path=continous_dataset_path)
    train_lda(dataset=dataset, continous_dataset_path=continous_dataset_path)


if __name__ == "__main__":
    main()