from typing import Tuple

import comet_ml
import datetime
import pytorch_lightning as pl

from settings import EXPERIMENT_NAME


def get_cometlogger() -> Tuple[pl.loggers.CometLogger, str]:
    experiment_creation_time = datetime.datetime.now().strftime("[%Y-%m-%d,%H:%M]")

    comet_logger = pl.loggers.CometLogger(
        api_key="3xX4JIrZCsKBpMeFSsbQBfh0W",
        project_name="bci-errp",
        workspace="ivopascal",
        experiment_name=EXPERIMENT_NAME,
        save_dir='logs/',
    )

    return comet_logger, experiment_creation_time


def perform_basic_logging(comet_logger, train_set, val_set, test_set, model):
    comet_logger.auto_output_logging = "simple"

    comet_logger.experiment.log_html(model.explain_model())

    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)
    total_size = train_size + val_size + test_size
    dataset_sizes_txt = "<h2>Dataset sizes</h2>"
    dataset_sizes_txt += """<p>Train size: {} <br>
                                Validation size: {} <br>
                                Test size: {} <br>
                                Total size: {}<br>
                                (Download the datasets used in the Assets tab)<br><br>
                                Method to split whole dataset into train, validation and test datasets:<br>
                                {}<br></p>""".format(train_size, val_size,
                                                     test_size,
                                                     total_size, "balanced")
    comet_logger.experiment.log_html(dataset_sizes_txt)
