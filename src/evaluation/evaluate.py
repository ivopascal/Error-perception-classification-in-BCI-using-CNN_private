import math

import pytorch_lightning as pl
from torchmetrics.functional import precision, f1_score, recall, specificity, stat_scores

from settings import CKPT_PATH
from src.Models.model_core import ModelCore
from src.data.Datamodule import DataModule


def evaluate_model(trainer: pl.Trainer, dm: DataModule, model: ModelCore, comet_logger: pl.loggers.CometLogger):
    trainer.test(model=model, datamodule=dm, ckpt_path=CKPT_PATH)
    (y_true, y_predicted) = model.get_test_labels_predictions()

    y_true = y_true.reshape(-1).clone().to('cpu')
    y_predicted = y_predicted.reshape(-1).clone().to('cpu')
    y_predicted[y_predicted > 0.5] = 1
    y_predicted[y_predicted <= 0.5] = 0

    y_true_matrix, y_predicted_matrix = [], []
    for i in range(len(y_true)):
        y_true_add = [0, 0]
        y_pred_add = [0, 0]
        y_true_add[int(y_true[i])] = 1
        y_pred_add[int(y_predicted[i])] = 1
        y_true_matrix.append(y_true_add)
        y_predicted_matrix.append(y_pred_add)

    comet_logger.experiment.log_confusion_matrix(title="Confusion matrix",
                                                 labels=["ErrP", "No ErrP"],
                                                 y_true=y_true_matrix,
                                                 y_predicted=y_predicted_matrix)

    tp, fp, tn, fn, support = stat_scores(y_predicted, y_true)

    comet_logger.experiment.log_metric("true_positives", tp)
    comet_logger.experiment.log_metric("true_negatives", tn)
    comet_logger.experiment.log_metric("false_positives", fp)
    comet_logger.experiment.log_metric("false_negatives", fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    n_mcc = (mcc + 1) / 2
    negative_predictive_value = tn / max(1, (tn + fn))
    accuracy_conf_matrix = (tp + tn) / (fp + fn + tp + tn)
    comet_logger.experiment.log_metric("specificity", specificity(y_predicted, y_true))
    comet_logger.experiment.log_metric("sensitivity", recall(y_predicted, y_true))
    comet_logger.experiment.log_metric("precision", precision(y_predicted, y_true))
    comet_logger.experiment.log_metric("negative_predictive_value", negative_predictive_value)
    comet_logger.experiment.log_metric("accuracy_conf_matrix", accuracy_conf_matrix)
    comet_logger.experiment.log_metric("F1_score", f1_score(y_predicted, y_true))
    comet_logger.experiment.log_metric("MCC", mcc)
    comet_logger.experiment.log_metric("nMCC",  n_mcc)
