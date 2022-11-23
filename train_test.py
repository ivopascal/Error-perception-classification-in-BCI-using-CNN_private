from datetime import datetime
from typing import Optional

from src.comet_logging.comet_logger import get_cometlogger, perform_basic_logging
import pytorch_lightning as pl
import time
import torch

from src.data.Datamodule import DataModule
from src.data.build_dataset import build_dataset
from src.evaluation.evaluate import evaluate_model
from settings import PROJECT_MODEL_SAVES_FOLDER, OVERRIDEN_HYPER_PARAMS, MODEL_CLASS, \
    EXPERIMENT_NAME, CKPT_PATH
from src.util.dataclasses import EpochedDataSet


DEBUG_MODE = False


def train(dataset_file_path: Optional[str] = None, dataset: Optional[EpochedDataSet] = None):
    train_set, val_set, test_set = build_dataset(dataset_file_path, dataset)

    experiment_creation_time = datetime.now().strftime("[%Y-%m-%d,%H:%M]")
    model = MODEL_CLASS(train_set, test_set, val_set, hyperparams=OVERRIDEN_HYPER_PARAMS)
    dm = DataModule(train_set, val_set, test_set, batch_size=model.hyper_params["batch_size"])
    comet_logger, _ = get_cometlogger()

    trainer = pl.Trainer(
        max_epochs=model.hyper_params['max_num_epochs'],
        logger=comet_logger,
        fast_dev_run=DEBUG_MODE,
        accelerator="mps",
        devices=1,
        precision=32,
        log_every_n_steps=30,
    )

    start_time = time.time()

    perform_basic_logging(comet_logger, train_set, val_set, test_set, model)

    trainer.fit(model, datamodule=dm)

    # Log training time (to Metric tab and HTML tab)
    training_time = round(time.time() - start_time, 1)
    comet_logger.experiment.log_metric("training_time_sec", training_time)
    train_time_txt = "<h2>Training duration</h2>"
    train_time_txt += "<p>{} seconds (~{} minutes)</p><br>".format(training_time, round(training_time / 60))
    comet_logger.experiment.log_html(train_time_txt)

    # Saves a serialized object to disk. This function uses Python’s pickle utility for serialization.
    model_save_path = PROJECT_MODEL_SAVES_FOLDER + EXPERIMENT_NAME + "_{}.pt".format(experiment_creation_time)
    torch.save(model.state_dict(), model_save_path)

    # Log the current model (provides downloadable link)
    comet_logger.experiment.log_model(EXPERIMENT_NAME, model_save_path)
    comet_logger.experiment.log_asset(EXPERIMENT_NAME, 'settings.py')

    evaluate_model(trainer, dm, model, comet_logger)

    comet_logger.experiment.end()


if __name__ == "__main__":
    train("/Users/ivopascal/Documents/PhD/Error-perception-classification-in-BCI-using-CNN/BCI_root/Datasets"
          "/Monitoring_error-related_potentials_2015/Datasets_pickle_files/Balanced/bp[low:1,high:10,ord:6]_epoch["
          "onset:0,size:600]_bal[added_to:0,#:3055].p")
