import math
from datetime import datetime
from typing import List

import torch
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from torchmetrics.functional import precision, f1_score, recall, stat_scores

from settings import CKPT_PATH, PROJECT_IMAGES_FOLDER, EXPERIMENT_NAME, PROJECT_RESULTS_FOLDER
from src.Models.model_core import ModelCore
from src.data.Datamodule import DataModule
from src.data.util import save_file_pickle
from src.util.dataclasses import EvaluationMetrics, StatScores
from sklearn.metrics import roc_curve, auc


def calculate_metrics_ensemble(trainer, models: List[ModelCore], datamodule, ckpt_path) -> EvaluationMetrics:
    all_y_predictions = []
    for model in models:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx = model.get_test_labels_predictions()
        all_y_predictions.append(y_predicted)

    all_y_predictions = torch.stack(all_y_predictions)
    y_predicted = all_y_predictions.mean(dim=0)
    y_variance = all_y_predictions.std(dim=0)

    return build_evaluation_metrics(y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx)


def calculate_metrics(trainer, model: ModelCore, datamodule, ckpt_path) -> EvaluationMetrics:
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx = model.get_test_labels_predictions()

    return build_evaluation_metrics(y_true, y_predicted, y_variance, y_in_distribution, y_subj_idx)


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

    tp, fp, tn, fn, support = stat_scores(binarized_y_predicted, y_true)
    try:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        n_mcc = (mcc + 1) / 2
    except:
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
        precision(binarized_y_predicted, y_true),
        recall(binarized_y_predicted, y_true),
        negative_predictive_value,
        accuracy_conf_matrix,
        f1_score(binarized_y_predicted, y_true),
        mcc,
        n_mcc,
    )


def plot_and_log_roc(evaluation_metrics, hyper_params, comet_logger):
    fpr, tpr, threshold = roc_curve(evaluation_metrics.y_true.numpy(),
                                    evaluation_metrics.y_predicted.numpy())
    roc_auc = auc(fpr, tpr)

    if hyper_params.get("bayesian_forward_passes"):
        plt.plot(fpr, tpr, label=f"AUC bayesian {hyper_params['bayesian_forward_passes']} = {roc_auc}")
    else:
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve on continuous domain. OoD labeled 0 class")
    plt.legend()

    if hyper_params.get("bayesian_forward_passes"):
        image_file_name = PROJECT_IMAGES_FOLDER + EXPERIMENT_NAME + \
                          f"-roc-curve-bayesian{hyper_params['bayesian_forward_passes']}.png"
    else:
        image_file_name = PROJECT_IMAGES_FOLDER + EXPERIMENT_NAME + "-roc-curve.png"

    plt.savefig(image_file_name)

    if comet_logger:
        comet_logger.experiment.log_metric("ROC_AUC", roc_auc)
        comet_logger.experiment.log_image(image_file_name)


def log_evaluation_metrics_to_comet(evaluation_metrics: EvaluationMetrics, comet_logger, prefix=""):
    log_metric = comet_logger.experiment.log_metric
    log_metric(f"{prefix}true_positives", evaluation_metrics.statscores.tp)
    log_metric(f"{prefix}true_negatives", evaluation_metrics.statscores.tn)
    log_metric(f"{prefix}false_positives", evaluation_metrics.statscores.fp)
    log_metric(f"{prefix}false_negatives", evaluation_metrics.statscores.fn)
    log_metric(f"{prefix}precision", evaluation_metrics.precision)
    log_metric(f"{prefix}recall", evaluation_metrics.recall)
    log_metric(f"{prefix}negative_predictive_value", evaluation_metrics.negative_predictive_value)
    log_metric(f"{prefix}accuracy_conf_matrix", evaluation_metrics.accuracy_conf_matrix)
    log_metric(f"{prefix}F1_score", evaluation_metrics.f1_score)
    log_metric(f"{prefix}MCC", evaluation_metrics.mcc)
    log_metric(f"{prefix}nMCC", evaluation_metrics.n_mcc)


def evaluate_model(trainer: pl.Trainer, dm: DataModule, model: ModelCore, comet_logger: pl.loggers.CometLogger):
    metrics = calculate_metrics(trainer, model, dm, ckpt_path=CKPT_PATH)
    save_file_pickle(metrics, PROJECT_RESULTS_FOLDER +
                     f"metrics_{EXPERIMENT_NAME}_{datetime.now().strftime('[%Y-%m-%d,%H:%M]')}.pkl")
    log_evaluation_metrics_to_comet(metrics, comet_logger)
