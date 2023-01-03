import torch
from torch import nn

from src.Models.EegNet import ProperEEGNet
from src.util.nn_modules import Permute, DepthwiseConv2d, SeparableConv2d, enable_dropout, \
    SamplingSoftmax, Softplus
from src.util.util import uncertainty


class TwoHeadTrainModel(nn.Module):
    def __init__(self, F1, D, sampling_rate, n_output_nodes):
        super().__init__()
        F2 = F1 * D

        self.trunc_model = nn.Sequential(
            Permute((0, 1, 3, 2)),
            # Layer 1
            nn.Conv2d(1, F1, (int(sampling_rate / 2), 1), padding=0),
            nn.BatchNorm2d(F1, False),
            DepthwiseConv2d(F1, depth_multiplier=D, bias=False, padding='valid'),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),

            SeparableConv2d(F2, F2, (1, 16), bias=False),
            nn.BatchNorm2d(F2, False),
            nn.ELU(),

            nn.AvgPool3d((1, F1, 1)),
            nn.Dropout(0.25),

            # FC Layer
            nn.Flatten(),  #

            nn.Linear(192, n_output_nodes),
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

    def forward(self, x):
        all_means = []
        all_variances = []
        for _ in range(self.forward_passes):
            enable_dropout(self)
            latent = self.trunc_model(x)
            means = self.mean_block(latent)
            variances = self.variance_block(latent)
            all_means.append(means)
            all_variances.append(variances)

        means = torch.stack(all_means)
        variances = torch.stack(all_variances)
        stds = self.preprocess_variance_output(variances)

        y_logits_mean = means.mean(dim=0)
        mixture_var = (stds.square() + means.square()).mean(axis=0) - y_logits_mean.square()
        mixture_var[mixture_var < 0.0] = 0.0

        y_logits_std_epi = means.std(dim=0)
        y_logits_std_ale = stds.mean(dim=0)

        sampling_softmax = SamplingSoftmax(num_samples=self.forward_passes)

        y_probs = sampling_softmax([y_logits_mean, y_logits_std_ale + y_logits_std_epi])
        y_probs_epi = sampling_softmax([y_logits_mean, y_logits_std_epi])
        y_probs_ale = sampling_softmax([y_logits_mean, y_logits_std_ale])

        ale_entropy = uncertainty(y_probs_ale.cpu().numpy())
        epi_entropy = uncertainty(y_probs_epi.cpu().numpy())

        return y_probs, ale_entropy, epi_entropy

    @staticmethod
    def preprocess_variance_output(variances):
        return variances


class DisentangledModel(ProperEEGNet):

    def create_model_architecture(self):
        self.train_model = TwoHeadTrainModel(self.get_hyperparams()['F1'],
                                             self.get_hyperparams()['D'],
                                             self.get_hyperparams()['sampling_rate'],
                                             self.get_n_output_nodes())
        self.predict_model = TwoHeadPredictModel(self.train_model, self.get_hyperparams()['two_head_passes'],)
        return self.train_model

    def test_step(self, batch, batch_idx):

        x, y_all = batch

        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        pred_mean, ale_uncertainty, epi_uncertainty = self.predict_model(x)

        disentangle_logs = {"pred_mean": pred_mean,
                            "ale_uncertainty": ale_uncertainty,
                            "epi_uncertaitny": epi_uncertainty}

        test_step_output = super().test_step(batch, batch_idx)

        return {**test_step_output, **disentangle_logs}

    def test_epoch_end(self, outputs):
        self.ale_uncertainty = torch.stack([x['ale_uncertainty'] for x in outputs])
        self.epi_uncertainty = torch.stack([x['epi_uncertainty'] for x in outputs])
        self.pred_mean = torch.stack([x['pred_mean'] for x in outputs])

        return super().test_epoch_end(outputs)

    def get_test_labels_predictions(self):
        return self.test_y_true, self.test_y_predicted, self.test_y_variance, self.test_y_in_distribution, self.test_y_subj_idx

    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["two_head_passes"] = 100
        hyper_params["num_classes"] = 2  # Probably doesn't do anything
        return hyper_params

    def get_n_output_nodes(self):
        return 2

    def get_loss_function(self):
        return nn.CrossEntropyLoss()
