from datetime import datetime

import pytorch_lightning as pl

from settings import DEBUG_MODE, PROJECT_RESULTS_FOLDER, EXPERIMENT_NAME, \
    CONTINUOUS_TESTING_INTERVAL
from src.data.Datamodule import ContinuousDataModule
from src.data.build_dataset import build_continuous_dataset
from src.data.util import save_file_pickle

from src.evaluation.evaluate import calculate_metrics, log_evaluation_metrics_to_comet, \
    log_continuous_metrics


def test_continuous(models, comet_logger, dataset_folder):
    if not isinstance(models, list):
        models = [models]

    train_set, val_set, test_set = build_continuous_dataset(dataset_folder)

    if not models:
        raise ValueError("Models needs to be provided")

    if DEBUG_MODE:
        test_set = test_set[0][:1], test_set[1][:1]

    dm = ContinuousDataModule(train_set, val_set, test_set,
                              batch_size=models[0].hyper_params["batch_size"],
                              interval=CONTINUOUS_TESTING_INTERVAL)

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        precision=32,
    )
    print("Testing against continuous data...")
    metrics = calculate_metrics(trainer, models, dm, ckpt_path=None)
    log_evaluation_metrics_to_comet(metrics, comet_logger, prefix="Continuous_")
    log_continuous_metrics(metrics, comet_logger)

    save_file_pickle(metrics, PROJECT_RESULTS_FOLDER +
                     f"metrics_{EXPERIMENT_NAME}_continuous_{datetime.now().strftime('[%Y-%m-%d,%H:%M]')}.pkl")
