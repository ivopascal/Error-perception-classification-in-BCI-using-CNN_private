from torch import nn
import torcheeg.models
from torcheeg.models.cnn.fbcnet import LinearWithConstraint

from src.Models.model_core import ModelCore


class NoSoftMaxFBCNet(torcheeg.models.FBCNet):
    def last_block(self, in_channels, out_channels, weight_norm=True):
        return nn.Sequential(
            LinearWithConstraint(in_channels,
                                 out_channels,
                                 max_norm=0.5,
                                 weight_norm=weight_norm))


class FBCNet(ModelCore):

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
            'num_S': 32,
            'temporal_layer': 'LogVarLayer',
            'stride_factor': 4,
        }

    def explain_model(self):
        return """
        An Efficient Multi-view Convolutional Neural Network for Brain-Computer Interface.
        Implemented by https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.models.FBCNet.html#torcheeg.models.FBCNet
        
        Designed by https://github.com/ravikiran-mane/FBCNet
        Published under https://arxiv.org/abs/2104.01233
        """

    def create_model_architecture(self):
        return nn.Sequential(
            NoSoftMaxFBCNet(num_electrodes=self.get_hyperparams()['input_size'][0],
                            chunk_size=self.get_hyperparams()['input_size'][1],
                            in_channels=1,
                            num_S=self.get_hyperparams()['num_S'],
                            num_classes=self.get_n_output_nodes(),
                            temporal=self.get_hyperparams()['temporal_layer'],
                            stride_factor=self.get_hyperparams()['stride_factor']
                            ),
            nn.Sigmoid()
        )

    def get_n_output_nodes(self):
        return 1

    def get_loss_function(self):
        return nn.BCELoss()
