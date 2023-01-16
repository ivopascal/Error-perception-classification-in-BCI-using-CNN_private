from datetime import datetime
from typing import Optional, cast

from pytorch_lightning.callbacks import EarlyStopping

from continuous import test_continuous
from src.Models.model_core import ModelCore
from src.comet_logging.comet_logger import get_cometlogger, perform_basic_logging
import pytorch_lightning as pl
import time
import torch
from pydoc import locate

from src.data.Datamodule import DataModule
from src.data.build_dataset import build_dataset
from src.evaluation.evaluate import evaluate_model, log_evaluation_metrics_to_comet, calculate_metrics
from settings import PROJECT_MODEL_SAVES_FOLDER, OVERRIDEN_HYPER_PARAMS, MODEL_CLASS_NAME, \
    EXPERIMENT_NAME, ENSEMBLE_SIZE, EARLY_STOPPING_PATIENCE
from src.util.dataclasses import EpochedDataSet


def train(dataset_file_path: Optional[str] = None,
          dataset: Optional[EpochedDataSet] = None,
          continuous_dataset_path=None):
    train_set, val_set, test_set = build_dataset(dataset_file_path, dataset)

    experiment_creation_time = datetime.now().strftime("[%Y-%m-%d,%H:%M]")
    model_class: ModelCore = cast(ModelCore, locate(f"src.Models.{MODEL_CLASS_NAME}"))
    models = []
    for _ in range(ENSEMBLE_SIZE):
        model = model_class(train_set, test_set, val_set, hyperparams=OVERRIDEN_HYPER_PARAMS)
        dm = DataModule(train_set, val_set, test_set, batch_size=model.hyper_params["batch_size"],
                        test_batch_size=model.hyper_params.get("test_batch_size"))
        comet_logger, _ = get_cometlogger()

        trainer = pl.Trainer(
            max_epochs=model.hyper_params['max_num_epochs'],
            # callbacks=[EarlyStopping(monitor="loss_val", min_delta=0.00, patience=EARLY_STOPPING_PATIENCE,
            #                          verbose=False, mode="min")],
            logger=comet_logger,
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

        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        comet_logger.experiment.log_metric("parameters", n_trainable_params)

        train_time_txt = "<h2>Training duration</h2>"
        train_time_txt += "<p>{} seconds (~{} minutes)</p><br>".format(training_time, round(training_time / 60))
        comet_logger.experiment.log_html(train_time_txt)

        # Saves a serialized object to disk. This function uses Python’s pickle utility for serialization.
        model_save_path = PROJECT_MODEL_SAVES_FOLDER + EXPERIMENT_NAME + "_{}.pt".format(experiment_creation_time)
        torch.save(model.state_dict(), model_save_path)

        # Log the current model (provides downloadable link)
        comet_logger.experiment.log_model(EXPERIMENT_NAME, model_save_path)
        comet_logger.experiment.log_code('settings.py')

        evaluate_model(trainer, dm, model, comet_logger)

        models.append(model)
        comet_logger.experiment.end()

    comet_logger, _ = get_cometlogger()
    metrics = calculate_metrics(trainer, models, dm, ckpt_path=None) # noqa
    log_evaluation_metrics_to_comet(metrics, comet_logger, prefix="Ensemble_")

    test_continuous(models=models, comet_logger=comet_logger, dataset_folder=continuous_dataset_path)


if __name__ == "__main__":
    train("/Users/ivopascal/Documents/PhD/Error-perception-classification-in-BCI-using-CNN/BCI_root/Datasets"
          "/Monitoring_error-related_potentials_2015/Datasets_pickle_files/Balanced/bp[low:1,high:10,ord:6]_epoch["
          "onset:0,size:600]_bal[added_to:0,#:3055].p")
