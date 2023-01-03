import torcheeg.models

from src.Models.model_core import ModelCore
import torch.nn as nn

from src.util.nn_modules import Permute, DepthwiseConv2d, SeparableConv2d, View, Squeeze


class EEGNet(ModelCore):
    # This is able to overfit to session 1, failing to generalise to session 2
    # Session 1 val = 0.95, but session 2 acc = 0.729

    def get_default_hyperparameters(self, test_dataset):
        return {
            'input_size': test_dataset[0][0].shape,
            'num_classes': 1,
            'batch_size': 120,
            'test_batch_size': 10,
            'max_num_epochs': 1200,
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'weight_decay': 0,
        }

    def explain_model(self):
        return """
        Implementation of EEGNet according to https://github.com/aliasvishnu/EEGNet/
        
        The original authors provide a tensorflow implementation here https://github.com/vlawhern/arl-eegmodels
        """

    def create_model_architecture(self):
        return nn.Sequential(
            Permute((0, 1, 3, 2)),
            # Layer 1
            nn.Conv2d(1, 16, (1, 64), padding=0),
            nn.ELU(),
            nn.BatchNorm2d(16, False),
            nn.Dropout(0.25),
            Permute((0, 3, 1, 2)),

            # Layer 2
            nn.ZeroPad2d((16, 17, 0, 1)),
            nn.Conv2d(1, 4, (2, 32)),
            nn.ELU(),
            nn.BatchNorm2d(4, False),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 4),

            # Layer 3
            nn.ZeroPad2d((2, 1, 3, 3)),  # This deviates from sourcecode
            nn.Conv2d(4, 4, (8, 4)),
            nn.ELU(),
            nn.BatchNorm2d(4, False),
            nn.Dropout(0.25),
            nn.MaxPool2d((2, 4)),

            # FC Layer
            nn.Flatten(),
            nn.Linear(76, 1),
            nn.Sigmoid(),
        )

    def get_n_output_nodes(self):
        return 1

    def get_loss_function(self):
        return nn.BCELoss()


class BayesianEEGNet(EEGNet):
    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["bayesian_forward_passes"] = 10  # 50 is the upper limit to stay at 512 Hz online
        hyper_params["test_batch_size"] = 100
        return hyper_params


class ProperEEGNet(EEGNet):
    def create_model_architecture(self):
        F1 = self.get_hyperparams()['F1']
        D = self.get_hyperparams()['D']
        sampling_rate = self.get_hyperparams()['sampling_rate']
        F2 = F1 * D

        return nn.Sequential(
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
            nn.Flatten(),
            nn.Linear(192, self.get_n_output_nodes()),
            nn.Sigmoid(),
        )

    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["F1"] = 8
        hyper_params["D"] = 2
        hyper_params["sampling_rate"] = 512

        return hyper_params


class TorchEEGNet(EEGNet):
    def create_model_architecture(self):
        F1 = self.get_hyperparams()['F1']
        D = self.get_hyperparams()['D']
        sampling_rate = self.get_hyperparams()['sampling_rate']
        F2 = F1 * D

        return nn.Sequential(
            torcheeg.models.EEGNet(chunk_size=self.get_hyperparams()['input_size'][1],
                                   num_electrodes=self.get_hyperparams()['input_size'][0],
                                   F1=F1,
                                   F2=F2,
                                   D=D,
                                   num_classes=self.get_n_output_nodes(),
                                   kernel_1=int(sampling_rate / 2),
                                   kernel_2=int(sampling_rate / 8)
                                   ),
            nn.Sigmoid()
        )

    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["F1"] = 8
        hyper_params["D"] = 2
        hyper_params["sampling_rate"] = 512

        return hyper_params

    def get_n_output_nodes(self):
        return 1


class LargeTorchEEGNet(EEGNet):
    def create_model_architecture(self):
        F1 = self.get_hyperparams()['F1']
        D = self.get_hyperparams()['D']
        sampling_rate = self.get_hyperparams()['sampling_rate']
        F2 = F1 * D

        return nn.Sequential(
            torcheeg.models.EEGNet(chunk_size=self.get_hyperparams()['input_size'][1],
                                   num_electrodes=self.get_hyperparams()['input_size'][0],
                                   F1=F1,
                                   F2=F2,
                                   D=D,
                                   num_classes=self.get_n_output_nodes(),
                                   kernel_1=int(sampling_rate / 2),
                                   kernel_2=int(sampling_rate / 8)
                                   ),
            nn.Sigmoid()
        )

    def get_default_hyperparameters(self, test_dataset):
        hyper_params = super().get_default_hyperparameters(test_dataset)
        hyper_params["F1"] = 16
        hyper_params["D"] = 4
        hyper_params["sampling_rate"] = 512

        return hyper_params

    def get_n_output_nodes(self):
        return 1
