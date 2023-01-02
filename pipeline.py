from preprocess import preprocess
from settings import MODEL_TYPE
from train_test import train
from train_test_lda import train_lda


def main():
    dataset, continous_dataset_path = preprocess()
    if MODEL_TYPE == "Pytorch":
        train(dataset=dataset, continuous_dataset_path=continous_dataset_path)
    elif MODEL_TYPE == "SKLearn":
        train_lda(dataset=dataset, continous_dataset_path=continous_dataset_path)


if __name__ == "__main__":
    main()
