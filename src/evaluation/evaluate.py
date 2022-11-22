import math

import pytorch_lightning as pl

from src.Models.model_core import ModelCore
from src.data.Datamodule import DataModule


def evaluate_model(trainer: pl.Trainer, dm: DataModule, model: ModelCore, comet_logger: pl.loggers.CometLogger):
    trainer.test(datamodule=dm, ckpt_path='best')
    (y_true, y_predicted) = model.get_test_labels_predictions()

    y_true = y_true.reshape(-1).tolist()
    y_true = [int(y == 4.0) for y in y_true]
    y_predicted = y_predicted.reshape(-1).tolist()

    y_true_matrix, y_predicted_matrix = [], []
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        y_true_add = [0, 0]
        y_pred_add = [0, 0]
        y_true_add[int(y_true[i])] = 1
        y_pred_add[int(y_predicted[i])] = 1
        y_true_matrix.append(y_true_add)
        y_predicted_matrix.append(y_pred_add)
        if y_true[i] == 0:
            if y_predicted[i] == 0:
                TP += 1
            else:
                FN += 1
        else:
            if y_predicted[i] == 0:
                FP += 1
            else:
                TN += 1

    comet_logger.experiment.log_confusion_matrix(title="Confusion matrix",
                                                 labels=["ErrP", "No ErrP"],
                                                 y_true=y_true_matrix,
                                                 y_predicted=y_predicted_matrix)

    comet_logger.experiment.log_metric("true_positives", TP)
    comet_logger.experiment.log_metric("true_negatives", TN)
    comet_logger.experiment.log_metric("false_positives", FP)
    comet_logger.experiment.log_metric("false_negatives", FN)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    negative_predictive_value = TN / (TN + FN)
    accuracy_conf_matrix = (TP + TN) / (FP + FN + TP + TN)
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    mcc = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    n_mcc = (mcc + 1) / 2
    comet_logger.experiment.log_metric("specificity", specificity)
    comet_logger.experiment.log_metric("sensitivity", sensitivity)
    comet_logger.experiment.log_metric("precision", precision)
    comet_logger.experiment.log_metric("negative_predictive_value", negative_predictive_value)
    comet_logger.experiment.log_metric("accuracy_conf_matrix", accuracy_conf_matrix)
    comet_logger.experiment.log_metric("F1_score", f1_score)
    comet_logger.experiment.log_metric("MCC", mcc)
    comet_logger.experiment.log_metric("nMCC", n_mcc)
