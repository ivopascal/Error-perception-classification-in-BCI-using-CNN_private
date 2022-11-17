import random
from abc import abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from util.metrics import binary_acc


class ModelCore(pl.LightningModule):

    def __init__(self, train_dataset, val_dataset, test_dataset, hyperparams=None):
        super(ModelCore, self).__init__()

        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

        self.hyper_params = self.get_default_hyperparameters(test_dataset)

        # Overwrite hyperparameters if given
        if hyperparams:
            for key, val in hyperparams.items():
                self.hyper_params[key] = val

        self.model = self.create_model_architecture()
        self.loss_function = self.get_loss_function()

    def get_hyperparams(self):
        return self.hyper_params

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    @abstractmethod
    def get_default_hyperparameters(self, test_dataset):
        raise NotImplementedError

    @abstractmethod
    def explain_model(self):
        return "<p>Model explanation not implemented</p>"

    @abstractmethod
    def create_model_architecture(self):
        raise NotImplementedError

    def forward(self, x):
        x = x.float()

        # Convert from [Batch, Channel, Length] to [Batch, Channel, Height, Width]
        # Do this in order to correctly apply 2D convolution
        shape = x.shape
        x = x.view(shape[0], 1, shape[1], shape[2])

        return self.model(x)

    def calculate_loss_and_accuracy(self, train_batch, name):
        x, y = train_batch
        y = y[:, 4]  # get only label
        y_logits = self.forward(x)

        loss = self.loss_function(y_logits, y)
        y_hat = torch.round(F.softmax(y_logits, dim=-1))
        acc = binary_acc(y_hat, y)

        logs = {
            f'loss_{name}': loss,
            f'acc_{name}': acc.clone().detach(),
        }

        self.log_dict(logs)
        return {
            **logs,
            'loss': loss,
            'log': logs
        }

    def training_step(self, train_batch, batch_idx):
        return self.calculate_loss_and_accuracy(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self.calculate_loss_and_accuracy(val_batch, "val")

    def test_step(self, batch, batch_idx):
        x, y_all = batch
        y = y_all[:, 4]  # get only label
        y_logits = self.forward(x)
        y_hat = torch.round(F.softmax(y_logits, dim=-1))
        acc = binary_acc(y_hat, y)

        # Specificy the subject from where this sample comes from:
        subj = y_all[:, 0].tolist()
        subj_str = 'acc_' + str(subj[0])

        log = {
            'acc': acc.clone().detach(),
            subj_str: acc.clone().detach()
        }

        output = {
            **log,
            'y_true': y.clone().detach(),
            'y_predicted': torch.tensor([int(x[0] == 0) for x in y_hat])  # convert list of pairs to y_prediction
        }
        self.log_dict(log)

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss_val'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc_val'] for x in outputs]).mean()

        # Define outputs
        logs = {'loss_val': avg_loss, 'acc_val': avg_acc}
        self.log_dict(logs)
        return {'val_loss': avg_loss, 'log': logs}

    def test_epoch_end(self, outputs):
        test_acc = torch.stack([x['acc'] for x in outputs]).mean()
        test_y_true = torch.stack([x['y_true'] for x in outputs])
        test_y_predicted = torch.stack([x['y_predicted'] for x in outputs])

        # Get accuracy per subject
        test_acc_subj = [0] * 6
        for i in range(6):
            subj_str = 'acc_' + str(i + 1)
            accuracies = [x[subj_str] for x in outputs if (subj_str in x)]
            if not accuracies:
                test_acc_subj[i] = 0
                test_acc_subj[i] = 0
            else:
                test_acc_subj[i] = torch.stack(accuracies).mean()

        logs = {'acc_test': test_acc}
        for i in range(6):
            subj_str = 'acc_subj_' + str(i + 1)
            logs[subj_str] = self.test_acc_subj[i]

        self.log_dict(logs)
        return {'log': logs}

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hyper_params['test_batch_size'], num_workers=8)

    def configure_optimizers(self):
        # Use Stochastic Gradient Descent
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hyper_params['learning_rate'],
                                    weight_decay=self.hyper_params['weight_decay'])
        return [optimizer]


