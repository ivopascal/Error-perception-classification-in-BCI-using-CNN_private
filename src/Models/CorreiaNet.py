import torch.nn as nn

from src.Models.model_core import ModelCore


class CorreiaNet(ModelCore):
    def get_default_hyperparameters(self, test_dataset):
        return {
            'input_size': test_dataset[0][0].shape,
            'num_classes': 2,
            'batch_size': 120,
            'test_batch_size': 1,
            'max_num_epochs': 1200,
            'optimizer': 'SGD',
            'learning_rate': 0.001,
            'weight_decay': 0.00001,  # L2 regularization
            'momentum': 0.9,
        }

    def explain_model(self):
        # This will be added to the HTML tab in Comet
        text = ""

        # Model name
        model_name = "_own11"
        text += "<h1>{}</h1>".format(model_name)

        # Model key points
        key_points = """
      <p>This model is an experiment.</p>
      """
        # Further explain the architecture
        text += "{}".format(key_points)

        return text

    def create_model_architecture(self):
        return nn.Sequential(
            nn.BatchNorm2d(1),
            # Block 1
            nn.Conv2d(1, 16, kernel_size=(1, 64), stride=(1, 20)),
            # Block 2
            nn.Conv2d(16, 16, kernel_size=(64, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # FC
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(208, self.get_n_output_nodes()),
        )

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def get_n_output_nodes(self):
        return 1
