from datetime import datetime

import torch
import pytorch_lightning as pl

from settings import PROJECT_MODEL_SAVES_FOLDER, DEBUG_MODE, PROJECT_RESULTS_FOLDER, EXPERIMENT_NAME, \
    CONTINUOUS_TESTING_INTERVAL
from src.Models.CorreiaNet import CorreiaNet
from src.data.Datamodule import ContinuousDataModule
from src.data.build_dataset import build_continuous_dataset
from src.data.util import save_file_pickle

from src.evaluation.evaluate import calculate_metrics, log_evaluation_metrics_to_comet

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

    dm = ContinuousDataModule(train_set, val_set, test_set,
                              batch_size=model.hyper_params["batch_size"],
                              interval=CONTINUOUS_TESTING_INTERVAL)

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        precision=32,
    )
    print("Testing against continuous data...")
    metrics = calculate_metrics(trainer, model, dm, ckpt_path=None)
    log_evaluation_metrics_to_comet(metrics, comet_logger, prefix="Continuous_")
    comet_logger.experiment.log_metric("Variance when correct ID",
                                       metrics.y_variance[metrics.y_predicted == metrics.y_true & metrics.y_in_distribution].mean())
    comet_logger.experiment.log_metric("Variance when incorrect ID",
                                       metrics.y_variance[metrics.y_predicted != metrics.y_true & metrics.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when ID",
                                       metrics.y_variance[metrics.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when OOD",
                                       metrics.y_variance[~metrics.y_in_distribution].mean())

    save_file_pickle(metrics, PROJECT_RESULTS_FOLDER +
                     f"metrics_{EXPERIMENT_NAME}_continuous_{datetime.now().strftime('[%Y-%m-%d,%H:%M]')}.pkl")


if __name__ == "__main__":
    test_continuous(model_path=MODEL_PATH)
