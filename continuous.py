import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from settings import PROJECT_MODEL_SAVES_FOLDER
from src.Models.CorreiaNet import CorreiaNet
from src.data.Datamodule import ContinuousDataModule
from src.data.build_dataset import build_continuous_dataset
from sklearn.metrics import roc_curve, auc

MODEL_NAME = "CNN_baseline_[2022-11-23,15:27].pt"
MODEL_PATH = PROJECT_MODEL_SAVES_FOLDER + MODEL_NAME
model_class = CorreiaNet

DATASET_FOLDER = "BCI_root/Datasets/Monitoring_error-related_potentials_2015/Datasets_pickle_files/Pre-processed/bp[" \
                 "low:1,high:10,ord:6]"


def test_continuous(model_path = None, model = None):
    train_set, val_set, test_set = build_continuous_dataset(DATASET_FOLDER)
    model = model_class(train_dataset=train_set, val_dataset=val_set, test_dataset=test_set)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


    dm = ContinuousDataModule(train_set, val_set, test_set, batch_size=model.hyper_params["batch_size"])

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        precision=32,
    )
    trainer.test(model, datamodule=dm)

    (y_true, y_predicted) = model.get_test_labels_predictions()

    y_true = y_true.reshape(-1)
    y_predicted = y_predicted.reshape(-1)

    fpr, tpr, threshold = roc_curve(y_true.cpu().numpy(), y_predicted.cpu().numpy())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve on continuous domain. All out-of-distribution samples are considered non-ErrP")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_continuous(model_path=MODEL_PATH)
