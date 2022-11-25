import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from settings import PROJECT_MODEL_SAVES_FOLDER, PROJECT_IMAGES_FOLDER, EXPERIMENT_NAME, DEBUG_MODE
from src.Models.CorreiaNet import CorreiaNet
from src.data.Datamodule import ContinuousDataModule
from src.data.build_dataset import build_continuous_dataset
from sklearn.metrics import roc_curve, auc

MODEL_NAME = "CNN_baseline_[2022-11-23,15:27].pt"
MODEL_PATH = PROJECT_MODEL_SAVES_FOLDER + MODEL_NAME
model_class = CorreiaNet

DATASET_FOLDER = "BCI_root/Datasets/Monitoring_error-related_potentials_2015/Datasets_pickle_files/Pre-processed/bp[" \
                 "low:1,high:10,ord:6]"  # This dataset folder isn't moving along yet


def test_continuous(model_path=None, model=None, comet_logger=None, dataset_folder=None):
    if not dataset_folder:
        dataset_folder = DATASET_FOLDER
    train_set, val_set, test_set = build_continuous_dataset(dataset_folder)

    if not model:
        if not model_path:
            raise ValueError("Either a model or a model_path needs to be supplied")
        model = model_class(train_dataset=train_set, val_dataset=val_set, test_dataset=test_set)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    if DEBUG_MODE:
        test_set = test_set[0][:1], test_set[1][:1]

    dm = ContinuousDataModule(train_set, val_set, test_set, batch_size=model.hyper_params["batch_size"])

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        precision=32,
    )
    print("Testing against continuous data...")
    trainer.test(model, datamodule=dm)

    (y_true, y_predicted) = model.get_test_labels_predictions()

    y_true = y_true.reshape(-1)
    y_predicted = y_predicted.reshape(-1)

    fpr, tpr, threshold = roc_curve(y_true.cpu().numpy(), y_predicted.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    if model.hyper_params.get("bayesian_forward_passes"):
        plt.plot(fpr, tpr, label=f"AUC bayesian {model.hyper_params['bayesian_forward_passes']} = {roc_auc}")
    else:
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve on continuous domain. OoD labeled 0 class")
    plt.legend()

    if model.hyper_params.get("bayesian_forward_passes"):
        image_file_name = PROJECT_IMAGES_FOLDER + EXPERIMENT_NAME + \
                          f"-roc-curve-bayesian{model.hyper_params['bayesian_forward_passes']}.png"
    else:
        image_file_name = PROJECT_IMAGES_FOLDER + EXPERIMENT_NAME + "-roc-curve.png"

    plt.savefig(image_file_name)

    if comet_logger:
        comet_logger.experiment.log_metric("Continuous ROC_AUC", roc_auc)
        comet_logger.experiment.log_image(image_file_name)


if __name__ == "__main__":
    test_continuous(model_path=MODEL_PATH)
