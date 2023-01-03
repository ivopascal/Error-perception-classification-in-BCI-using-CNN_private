import math
from datetime import datetime
from typing import List, Union

import torch
import pytorch_lightning as pl
from torchmetrics.functional import precision, f1_score, recall, stat_scores, specificity, accuracy

from settings import CKPT_PATH, EXPERIMENT_NAME, PROJECT_RESULTS_FOLDER
from src.Models.model_core import ModelCore
from src.data.Datamodule import DataModule
from src.data.util import save_file_pickle
from src.evaluation.per_participant import split_metrics_per_participant
from src.plots.average_around_event import plot_average_around_event
from src.plots.prediction_variance import plot_prediction_variance
from src.plots.roc_auc import plot_roc_auc
from src.util.dataclasses import EvaluationMetrics, StatScores


def calculate_metrics(trainer, models: Union[ModelCore, List[ModelCore]], datamodule, ckpt_path) -> EvaluationMetrics:
    if not isinstance(models, list):
        models = [models]
    all_y_predictions = []

    for model in models:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx = model.get_test_labels_predictions()
        all_y_predictions.append(y_predicted)

    all_y_predictions = torch.stack(all_y_predictions).type(dtype=torch.float32)
    y_predicted = all_y_predictions.mean(dim=0)
    if len(models) > 1:
        y_variance = all_y_predictions.std(dim=0)

    return build_evaluation_metrics(y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx)  # noqa


def build_evaluation_metrics(y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx) -> EvaluationMetrics:
    y_true = y_true.reshape(-1).clone().to('cpu')
    y_variance = y_variance.reshape(-1).clone().to('cpu')
    y_in_distribution = y_in_distribution.reshape(-1).clone().to('cpu')
    y_predicted = y_predicted.reshape(-1).clone().to('cpu')
    y_subj_idx = y_subj_idx.reshape(-1).clone().to('cpu')
    binarized_y_predicted = y_predicted.clone()
    binarized_y_predicted[binarized_y_predicted > 0.5] = 1
    binarized_y_predicted[binarized_y_predicted <= 0.5] = 0

    y_true_matrix, y_predicted_matrix = [], []
    for i in range(len(y_true)):
        y_true_add = [0, 0]
        y_pred_add = [0, 0]
        y_true_add[int(y_true[i])] = 1
        y_pred_add[int(binarized_y_predicted[i])] = 1
        y_true_matrix.append(y_true_add)
        y_predicted_matrix.append(y_pred_add)

    tp, fp, tn, fn, support = stat_scores(binarized_y_predicted, y_true, task="binary")
    try:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        n_mcc = (mcc + 1) / 2
    except Exception:
        mcc = -1
        n_mcc = -1
    negative_predictive_value = tn / max(1, (tn + fn))
    accuracy_conf_matrix = (tp + tn) / (fp + fn + tp + tn)

    return EvaluationMetrics(
        y_true,
        y_predicted,
        y_variance,
        y_in_distribution,
        y_subj_idx,
        y_true_matrix,
        y_predicted_matrix,
        StatScores(tp, fp, tn, fn, support),
        precision(binarized_y_predicted, y_true, task="binary"),
        specificity(binarized_y_predicted, y_true, task="binary"),
        accuracy(binarized_y_predicted, y_true, task="binary"),
        recall(binarized_y_predicted, y_true, task="binary"),
        negative_predictive_value,
        accuracy_conf_matrix,
        f1_score(binarized_y_predicted, y_true, task="binary"),
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


def log_continuous_metrics(metrics, comet_logger):
    comet_logger.experiment.log_metric("Variance when correct ID",
                                       metrics.y_variance[metrics.y_predicted == metrics.y_true
                                                          & metrics.y_in_distribution].mean())
    comet_logger.experiment.log_metric("Variance when incorrect ID",
                                       metrics.y_variance[metrics.y_predicted != metrics.y_true
                                                          & metrics.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when ID",
                                       metrics.y_variance[metrics.y_in_distribution].mean())

    comet_logger.experiment.log_metric("Variance when OOD",
                                       metrics.y_variance[~metrics.y_in_distribution].mean())

    fig, ax = plot_average_around_event(metrics, -600, 1000, y_to_plot=0, label_prefix="non-error_")
    fig, ax = plot_average_around_event(metrics, -600, 1000, y_to_plot=1, label_prefix="error_", figax=(fig, ax))
    comet_logger.experiment.log_figure(figure_name="Average around event", figure=fig)

    fig, axs = plot_prediction_variance(metrics)
    comet_logger.experiment.log_figure(figure_name="Prediction vs variance", figure=fig)

    per_participants = split_metrics_per_participant(metrics)
    fig, ax = plot_roc_auc(per_participants)
    comet_logger.experiment.log_figure(figure_name="Per participant ROC", figure=fig)
