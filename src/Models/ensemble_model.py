from typing import List

import pytorch_lightning as pl
import torch

from src.Models.model_core import ModelCore
from src.util.dataclasses import PredLabels


def ensemble_predictions_variance(trainer: pl.Trainer,
                                  models: List[ModelCore],
                                  datamodule: pl.LightningDataModule) -> PredLabels:

    all_y_predictions = []
    for model in models:
        trainer.test(model=model, datamodule=datamodule)
        pred_labels = model.get_test_labels_predictions()
        all_y_predictions.append(pred_labels.y_predicted)

    all_y_predictions = torch.stack(all_y_predictions).type(dtype=torch.float32)
    y_predicted = all_y_predictions.mean(dim=0)
    if len(models) > 1:
        y_variance = all_y_predictions.std(dim=0)
    else:
        y_variance = pred_labels.y_variance

    pred_labels = PredLabels(pred_labels.y_true,
                             y_predicted,
                             y_variance,
                             pred_labels.y_epi_uncertainty,
                             pred_labels.y_ale_uncertainty,
                             pred_labels.y_in_distribution,
                             pred_labels.y_subj_idx,
                             )

    return pred_labels

