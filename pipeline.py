from preprocess import preprocess
from train_test import train


def main():
    preprocessed_file_path = preprocess()
    train(preprocessed_file_path)


if __name__ == "__main__":
    main()