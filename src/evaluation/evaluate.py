import math
from datetime import datetime
from typing import List, Union

import pytorch_lightning as pl
from torchmetrics.functional import precision, f1_score, recall, stat_scores, specificity, accuracy

from settings import CKPT_PATH, EXPERIMENT_NAME, PROJECT_RESULTS_FOLDER
from src.Models.disentangled import DisentangledModel, DisentangledEnsemble
from src.evaluation.ensembling import ensemble_predictions_variance
from src.Models.model_core import ModelCore
from src.data.Datamodule import DataModule
from src.data.util import save_file_pickle
from src.evaluation.per_participant import split_metrics_per_participant
from src.plots.average_around_event import plot_average_around_event
from src.plots.prediction_variance import plot_prediction_variance
from src.plots.roc_auc import plot_roc_auc
from src.util.dataclasses import EvaluationMetrics, StatScores, PredLabels


def calculate_metrics(trainer: pl.Trainer, models: Union[ModelCore, List[ModelCore]], datamodule, ckpt_path) -> EvaluationMetrics:
    if not isinstance(models, list):
        trainer.test(model=models, datamodule=datamodule)
        return build_evaluation_metrics(models.get_test_labels_predictions())

    if isinstance(models[0], DisentangledModel):

        model = DisentangledEnsemble([model.train_model for model in models], models[0])
        trainer.test(model=model, datamodule=datamodule)
        return build_evaluation_metrics(model.get_test_labels_predictions())

    return build_evaluation_metrics(ensemble_predictions_variance(trainer, models, datamodule))


def build_evaluation_metrics(pred_labels: PredLabels) -> EvaluationMetrics:
    binarized_y_predicted = pred_labels.y_predicted.clone()
    binarized_y_predicted[binarized_y_predicted > 0.5] = 1
    binarized_y_predicted[binarized_y_predicted <= 0.5] = 0

    y_true_matrix, y_predicted_matrix = [], []
    for i in range(len(pred_labels.y_true)):
        y_true_add = [0, 0]
        y_pred_add = [0, 0]
        y_true_add[int(pred_labels.y_true[i])] = 1
        y_pred_add[int(binarized_y_predicted[i])] = 1
        y_true_matrix.append(y_true_add)
        y_predicted_matrix.append(y_pred_add)

    tp, fp, tn, fn, support = stat_scores(binarized_y_predicted, pred_labels.y_true, task="binary")
    try:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        n_mcc = (mcc + 1) / 2
    except Exception:
        mcc = -1
        n_mcc = -1
    negative_predictive_value = tn / max(1, (tn + fn))
    accuracy_conf_matrix = (tp + tn) / (fp + fn + tp + tn)

    return EvaluationMetrics(
        pred_labels,
        y_true_matrix,
        y_predicted_matrix,
        StatScores(tp, fp, tn, fn, support),
        precision(binarized_y_predicted, pred_labels.y_true, task="binary"),
        specificity(binarized_y_predicted, pred_labels.y_true, task="binary"),
        accuracy(binarized_y_predicted, pred_labels.y_true, task="binary"),
        recall(binarized_y_predicted, pred_labels.y_true, task="binary"),
        negative_predictive_value,
        accuracy_conf_matrix,
        f1_score(binarized_y_predicted, pred_labels.y_true, task="binary"),
        mcc,
        n_mcc,
    )


def log_evaluation_metrics_to_comet(evaluation_metrics: EvaluationMetrics,
                                    comet_logger: pl.loggers.CometLogger, # noqa
                                    prefix=""):
    log_metric = comet_logger.experiment.log_metric
    log_metric(f"{prefix}true_positives", evaluation_metrics.statscores.tp)
    log_metric(f"{prefix}true_negatives", evaluation_metrics.statscores.tn)
    log_metric(f"{prefix}false_positives", evaluation_metrics.statscores.fp)
    log_metric(f"{prefix}false_negatives", evaluation_metrics.statscores.fn)
    log_metric(f"{prefix}precision", evaluation_metrics.precision)
    log_metric(f"{prefix}specificity", evaluation_metrics.specificity)
    log_metric(f"{prefix}accuracy", evaluation_metrics.accuracy)

    log_metric(f"{prefix}recall", evaluation_metrics.recall)
    log_metric(f"{prefix}negative_predictive_value", evaluation_metrics.negative_predictive_value)
    log_metric(f"{prefix}accuracy_conf_matrix", evaluation_metrics.accuracy_conf_matrix)
    log_metric(f"{prefix}F1_score", evaluation_metrics.f1_score)
    log_metric(f"{prefix}MCC", evaluation_metrics.mcc)
    log_metric(f"{prefix}nMCC", evaluation_metrics.n_mcc)

    comet_logger.experiment.log_confusion_matrix(y_true=evaluation_metrics.y_true_matrix,
                                                 y_predicted=evaluation_metrics.y_predicted_matrix,
                                                 title={prefix})


def evaluate_model(trainer: pl.Trainer, dm: DataModule, model: ModelCore, comet_logger: pl.loggers.CometLogger): # noqa
    metrics = calculate_metrics(trainer, model, dm, ckpt_path=CKPT_PATH)
    save_file_pickle(metrics, PROJECT_RESULTS_FOLDER +
                     f"metrics_{EXPERIMENT_NAME}_{datetime.now().strftime('[%Y-%m-%d,%H:%M]')}.pkl")
    log_evaluation_metrics_to_comet(metrics, comet_logger)


def log_continuous_metrics(pred_labels: PredLabels, comet_logger):
    comet_logger.experiment.log_metric("Variance when correct ID",
                                       pred_labels.y_variance[pred_labels.y_predicted == pred_labels.y_true
                                                          & pred_labels.y_in_distribution].mean())
    comet_logger.experiment.log_metric("Variance when incorrect ID",
                                       pred_labels.y_variance[pred_labels.y_predicted != pred_labels.y_true
                                                          & pred_labels.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when ID",
                                       pred_labels.y_variance[pred_labels.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when OOD",
                                       pred_labels.y_variance[~pred_labels.y_in_distribution].mean())

    fig, ax = plot_average_around_event(pred_labels, -600, 1000, y_to_plot=0, label_prefix="non-error_")
    fig, ax = plot_average_around_event(pred_labels, -600, 1000, y_to_plot=1, label_prefix="error_", figax=(fig, ax))
    comet_logger.experiment.log_figure(figure_name="Average around event", figure=fig)

    fig, axs = plot_prediction_variance(pred_labels)
    comet_logger.experiment.log_figure(figure_name="Prediction vs variance", figure=fig)

    per_participants = split_metrics_per_participant(pred_labels)
    fig, ax = plot_roc_auc(per_participants)
    comet_logger.experiment.log_figure(figure_name="Per participant ROC", figure=fig)
