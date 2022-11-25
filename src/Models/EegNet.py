from src.Models.model_core import ModelCore
import torch.nn as nn

from src.util.nn_modules import Permute


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
