import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import random

from Models.model_core import ModelCore


class ConvNet2C(ModelCore):

    def get_default_hyperparameters(self, test_dataset):
        return {
            'input_size': test_dataset[0][0].shape,
            'num_classes': 2,
            'batch_size': 120,
            'test_batch_size': 1,
            'max_num_epochs': 1200,
            'optimizer': 'SGD',  # SGD, Adam, ...
            'learning_rate': 0.0001,  # Learning rate for Optimizer
            # 'momentum': 0.9,                 # Momentum for Optimizer
            'weight_decay': 0.002,  # Weight decay (L2 regularization)
        }

    @staticmethod
    def get_model_name():
        return "ConvNet 2 channel"

    @staticmethod
    def get_keypoints_html_addendum():
        return "<p>This specific implementation assumes the 2 channel situation of FCz and Cz</p>"

    def explain_model(self):
        # This will be added to the HTML tab in Comet
        text = ""

        # Model name
        model_name = self.get_model_name()
        text += "<h1>{}</h1>".format(model_name)

        # Model key points
        key_points = """
    <p>This model is taken from the paper:<br>
    <i>J. M. M. Torres, T. Clarkson, E. A. Stepanov, C. C. Luhmann, M. D. Lerner, and G. Riccardi, “Enhanced Error Decoding from Error-Related Potentials using Convolutional Neural Networks,” Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS, vol. 2018-July, pp. 360–363, 2018.</i></p>

    <p>It defines a 4 layers CNN architecture.</p>
    """
        key_points += self.get_keypoints_html_addendum()

        # Further explain the architecture
        text += "{}".format(key_points)

        return text

    def create_model_architecture(self):
        return nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=(2, 20)),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 2)),
                nn.Conv2d(128, 64, kernel_size=(1, 10)),
                nn.ELU(),
                # paper describes 1x1 pooling but this wouldn't do anything
                # nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
                nn.Flatten(),
                nn.Linear(64 * 12, 2),  # 768 for 60 samples
        )


class ConvNet64C(ConvNet2C):
    def create_model_architecture(self):
        return nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=(20, 20)),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2)),
                nn.Conv2d(128, 64, kernel_size=(10, 10)),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Flatten(),
                nn.Linear(25344, 2),  # 25344 for 600ms, 2304 for 64 samples
        )

    @staticmethod
    def get_model_name():
        return "ConvNet 64 Channel"

    @staticmethod
    def get_keypoints_html_addendum():
        return "<p>This specific implementation assumes the 2 channel situation of FCz and Cz</p>"

