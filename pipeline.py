from preprocess import preprocess
from train_test import train


def main():
    dataset = preprocess()
    train(dataset=dataset)


if __name__ == "__main__":
    main()