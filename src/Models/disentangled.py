from typing import List

import torch
import torcheeg.models
from torch import nn

from settings import LOG_DISENTANGLED_UNCERTAINTIES_ON
from src.Models.EegNet import ProperEEGNet
from src.Models.model_core import ModelCore
from src.util.dataclasses import PredLabels
from src.util.nn_modules import enable_dropout, \
    SamplingSoftmax, Softplus
from src.util.util import uncertainty


class TwoHeadTrainModel(nn.Module):
    def __init__(self, F1, D, sampling_rate, n_output_nodes, chunk_size, num_electrodes):
        super().__init__()
        F2 = F1 * D

        self.trunc_model = nn.Sequential(
            torcheeg.models.EEGNet(chunk_size=chunk_size,
                                   num_electrodes=num_electrodes,
                                   F1=F1,
                                   F2=F2,
                                   D=D,
                                   num_classes=n_output_nodes,
                                   kernel_1=int(sampling_rate / 2),
                                   kernel_2=int(sampling_rate / 8)
                                   ),
            nn.Softmax(),
        )

        self.mean_block = nn.Linear(n_output_nodes, n_output_nodes)
        self.variance_block = nn.Sequential(
            nn.Linear(n_output_nodes, n_output_nodes),
            Softplus()  # Custom Softplus because torch doesn't work on mps
        )

        self.sampling_softmax = SamplingSoftmax(num_samples=100, variance_type="linear_std")

    def forward(self, x):
        latent = self.trunc_model(x)
        mean = self.mean_block(latent)
        var = self.variance_block(latent)  # softplus activation
        out = self.sampling_softmax([mean, var])

        return out


class TwoHeadPredictModel(nn.Module):
    def __init__(self, two_head_train_model, forward_passes):
        super().__init__()
        self.trunc_model = two_head_train_model.trunc_model
        self.mean_block = two_head_train_model.mean_block
        self.variance_block = two_head_train_model.variance_block
        self.forward_passes = forward_passes

    def _get_all_means_variances(self, x):
        all_means = []
        all_variances = []
        for _ in range(self.forward_passes):
            enable_dropout(self)
            latent = self.trunc_model(x)
            means = self.mean_block(latent)
            variances = self.variance_block(latent)
            all_means.append(means)
            all_variances.append(variances)

        return torch.stack(all_means), torch.stack(all_variances)

    def forward(self, x):
        means, variances = self._get_all_means_variances(x)

        stds = self.preprocess_variance_output(variances)

        y_logits_mean = means.mean(dim=0)

        y_logits_std_epi = means.std(dim=0)
        y_logits_std_ale = stds.mean(dim=0)

        sampling_softmax = SamplingSoftmax(num_samples=self.forward_passes)

        y_probs = sampling_softmax([y_logits_mean, y_logits_std_ale + y_logits_std_epi])
        y_probs_epi = sampling_softmax([y_logits_mean, y_logits_std_epi])
        y_probs_ale = sampling_softmax([y_logits_mean, y_logits_std_ale])

        ale_entropy = torch.Tensor(uncertainty(y_probs_ale.detach().cpu().numpy()))
        epi_entropy = torch.Tensor(uncertainty(y_probs_epi.detach().cpu().numpy()))

        return y_probs, ale_entropy, epi_entropy

    @staticmethod
    def preprocess_variance_output(variances):
        return variances


class DisentangledModel(ProperEEGNet):

    def create_model_architecture(self):
        self.train_model = TwoHeadTrainModel(self.get_hyperparams()['F1'],
                                             self.get_hyperparams()['D'],
                                             self.get_hyperparams()['sampling_rate'],
                                             self.get_n_output_nodes(),
                                             self.get_hyperparams()['input_size'][1],
                                             self.get_hyperparams()['input_size'][0])
        self.predict_model = TwoHeadPredictModel(self.train_model, self.get_hyperparams()['two_head_passes'], )
        return self.train_model

    def _predict_disentangled_uncertainties(self, batch):
        x, y_all = batch

        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        return self.predict_model(x)

    def test_step(self, batch, batch_idx):
        pred_mean, ale_uncertainty, epi_uncertainty = self._predict_disentangled_uncertainties(batch)

        disentangle_logs = {"pred_mean": pred_mean,
                            "ale_uncertainty": ale_uncertainty,
                            "epi_uncertainty": epi_uncertainty}

        test_step_output = super().test_step(batch, batch_idx)

        return {**test_step_output, **disentangle_logs}

    def calculate_loss_and_accuracy(self, batch, name):
        if name in LOG_DISENTANGLED_UNCERTAINTIES_ON:
            pred_mean, ale_uncertainty, epi_uncertainty = self._predict_disentangled_uncertainties(batch)

            self.log_dict({
                f'ale_uncertainty_{name}': ale_uncertainty.mean(),
                f'epi_uncertainty_{name}': epi_uncertainty.mean(),
            })

        return super().calculate_loss_and_accuracy(batch, name)

    def test_epoch_end(self, outputs):
        self.ale_uncertainty = torch.stack([x['ale_uncertainty'] for x in outputs])
        self.epi_uncertainty = torch.stack([x['epi_uncertainty'] for x in outputs])
        self.pred_mean = torch.stack([x['pred_mean'] for x in outputs])

        return super().test_epoch_end(outputs)

    def get_test_labels_predictions(self):
        return PredLabels(self.test_y_true,
                          self.test_y_predicted,
                          self.test_y_variance,
                          self.epi_uncertainty,
                          self.ale_uncertainty,
                          self.test_y_in_distribution,
                          self.test_y_subj_idx)

    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["two_head_passes"] = 100
        hyper_params["num_classes"] = 2  # Probably doesn't do anything
        return hyper_params

    def get_n_output_nodes(self):
        return 2

    def get_loss_function(self):
        return nn.CrossEntropyLoss()


class TwoHeadEnsembleTrainModel(nn.Module):

    def __init__(self, all_train_models):
        super().__init__()
        self.all_train_models = all_train_models
        self.sampling_softmax = SamplingSoftmax(num_samples=100, variance_type="linear_std")

    def forward(self, x):
        all_means = []
        all_variances = []
        for train_model in self.all_train_models:
            latent = train_model.trunc_model(x)
            means = train_model.mean_block(latent)
            variances = train_model.variance_block(latent)
            all_means.append(means)
            all_variances.append(variances)

        mean = torch.stack(all_means).mean(dim=0)
        var = torch.stack(all_variances).mean(dim=0)

        return self.sampling_softmax([mean, var])


class TwoHeadEnsemblePredictModel(TwoHeadPredictModel):

    def __init__(self, all_train_models):
        super().__init__(all_train_models[0], 0)
        self.all_train_models = all_train_models

    def _get_all_means_variances(self, x):
        all_means = []
        all_variances = []
        for train_model in self.all_train_models:
            latent = train_model.trunc_model(x)
            means = train_model.mean_block(latent)
            variances = train_model.variance_block(latent)
            all_means.append(means)
            all_variances.append(variances)

        return torch.stack(all_means), torch.stack(all_variances)


class DisentangledEnsemble(DisentangledModel):

    def __init__(self, trained_models: List[TwoHeadTrainModel], sample_modelcore: ModelCore):
        self.all_train_models = trained_models
        super().__init__(train_dataset=sample_modelcore.train_dataset,
                         val_dataset=sample_modelcore.val_dataset,
                         test_dataset=sample_modelcore.test_dataset)

    def create_model_architecture(self):
        self.all_train_models = nn.ModuleList(self.all_train_models)
        self.train_model = TwoHeadEnsembleTrainModel(self.all_train_models)
        self.predict_model = TwoHeadEnsemblePredictModel(self.all_train_models)

        return self.all_train_models[0]  # Needs to be some kind of averaging


