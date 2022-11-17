from typing import Tuple

import comet_ml
import datetime
import pytorch_lightning as pl
import pickle as pk


def get_cometlogger() -> Tuple[pl.loggers.CometLogger, str]:
    experiment_name = 'CNN_baseline'
    experiment_creation_time = datetime.datetime.now().strftime("[%Y-%m-%d,%H:%M]")

    experiment = comet_ml.Experiment(
        api_key="3xX4JIrZCsKBpMeFSsbQBfh0W",
        project_name="bci-errp",
        workspace="ivopascal",
    )

    comet_logger = pl.loggers.CometLogger(
        api_key="3xX4JIrZCsKBpMeFSsbQBfh0W",
        project_name="bci-errp",
        workspace="ivopascal",
        experiment_name=experiment_name,
        save_dir='logs/',
    )

    return comet_logger, experiment_creation_time


def perform_basic_logging(comet_logger, train_set, val_set, test_set, model):
    comet_logger.auto_output_logging = "simple"
    # train_file = open("train_file.p", 'wb')
    # val_file = open("val_file.p", 'wb')
    # test_file = open("test_file.p", 'wb')
    # pk.dump(train_set, train_file)
    # pk.dump(val_set, val_file)
    # pk.dump(test_set, test_file)
    # comet_logger.experiment.log_asset(file_data="train_file.p", file_name="train_file.p", overwrite=True)
    # comet_logger.experiment.log_asset(file_data="val_file.p", file_name="val_file.p", overwrite=True)
    # comet_logger.experiment.log_asset(file_data="test_file.p", file_name="test_file.p", overwrite=True)
    # train_file.close()
    # val_file.close()
    # test_file.close()

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
