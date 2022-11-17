import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split


class Net(pl.LightningModule):

    def explainModel():
        # Explain the model using HTML text
        # This will be added to the HTML tab in Comet
        text = ""

        # Model name
        model_name = "CNN-Luo"
        text += "<h1>{}</h1>".format(model_name)

        # Model key points
        key_points = """
    <p>This model is taken from the paper:<br>
    <i>T. J. Luo, Y. C. Fan, J. T. Lv, and C. Le Zhou, “Deep reinforcement learning from error-related potentials via an EEG-based brain-computer interface,” in Proceedings - 2018 IEEE International Conference on Bioinformatics and Biomedicine, BIBM 2018, 2018, pp. 697–701.</i></p>
    
    <p>It defines a 5 layers CNN architecture.</p>
    """
        # Further explain the architecture
        text += "{}".format(key_points)

        return text

    # Log hyperparameters in Comet
    def get_hyperparams(self):
        return self.hyper_params

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++ Define Architecture +++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __init__(self, input_size=None, hyperparams={}):
        super(Net, self).__init__()

        # Default hyper-parameters from original work
        self.hyper_params = {
            'input_size': input_size,
            'batch_size': 64,  # Batch size (try powers of 2)
            'test_batch_size': 1,
            'max_num_epochs': 500,
            'optimizer': 'SGD',  # SGD, Adam, ...
            'weight_decay': 0.00001,  # Weight decay (L2 regularization)
            'momentum': 0.9,  # Momentum for Optimizer
            'learning_rate': 1e-3,  # Learning rate for Optimizer
        }

        # Overwrite hyperparameters if given
        if (hyperparams):
            for key, val in hyperparams.items():
                self.hyper_params[key] = val

        # **************** Declare layers ****************
        self.layer1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 40, kernel_size=(1, 25)),
            nn.ELU(),
        )

        self.layer2 = nn.Sequential(
            # Spatial convolution
            nn.Conv2d(40, 40, kernel_size=(64, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
        )

        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
        )

        self.flatten = nn.Flatten()

        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(560, 2),
        )

        # +++++++++++++++++++++++++ Loss function +++++++++++++++++++++++++
        # self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convert input to float (match network weights type)
        x = x.float()

        # Convert from [Batch, Channel, Length] to [Batch, Channel, Height, Width]
        # Do this in order to correctly apply 2D convolution
        # Get current size format
        s = x.shape
        # Convert
        x = x.view(s[0], 1, s[1], s[2])

        # apply Batch Normalization to input
        out1 = self.layer1(x)
        # print("out1:{}\n".format(out1.shape))
        # [64, 40, 64, 283]
        out2 = self.layer2(out1)
        # print("out2:{}\n".format(out2.shape))
        # [64, 40, 1, 283]
        out3 = self.layer3(out2)
        # print("out3:{}\n".format(out3.shape))
        # [64, 40, 1, 14]
        out3_flat = self.flatten(out3)
        # print("out3f:{}\n".format(out3_flat.shape))
        # [64, 560]
        out4 = self.layer4(out3_flat)
        # print("out4:{}\n".format(out4.shape))
        # [64, 1]

        # print("\nSizes:\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
        #   x.shape,
        #   out0.shape,
        #   out1.shape,
        #   out2.shape,
        #   out3.shape,
        #   out4.shape,
        #   out5.shape))

        # Return output
        return out4

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++ Test, Validation and Test steps +++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---------- Train ----------
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y[:, 4]  # get only label

        # output: 1 node (logit. To classify first pass through sigmoid and use 0.5 as threshold [done in binary_acc])
        # y_logits = self.forward(x)
        # y_logits = y_logits.view(y_logits.shape[0])
        # y = y.type_as(y_logits)
        # loss = self.loss_function(y_logits, y)
        # acc = self.binary_acc(y_logits, y)

        # output: 2 nodes
        y_logits = self.forward(x)
        loss = self.loss_function(y_logits, y)
        y_hot = torch.round(F.softmax(y_logits, dim=-1))
        acc = self.binary_acc(y_hot, y)

        # Define outputs
        logs = {
            'loss_train': loss,
            'acc_train': acc.clone().detach()}
        output = {
            'loss_train': loss,
            'acc_train': acc.clone().detach(),
            'loss': loss,
            'log': logs
        }

        # print("Output train: {}".format(output))
        return output

    # ---------- Validation ----------
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y[:, 4]  # get only label

        # # 1 node output
        # y_logits = self.forward(x)
        # y_logits = y_logits.view(y_logits.shape[0])
        # y = y.type_as(y_logits)

        # loss = self.loss_function(y_logits, y)
        # acc = self.binary_acc(y_logits, y)

        # output: 2 nodes
        y_logits = self.forward(x)
        loss = self.loss_function(y_logits, y)
        y_hot = torch.round(F.softmax(y_logits, dim=-1))
        acc = self.binary_acc(y_hot, y)

        # Define outputs
        output = {
            'loss_val': loss,
            'acc_val': acc.clone().detach()
        }

        # print("Output val: {}".format(output))
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss_val'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc_val'] for x in outputs]).mean()

        # Define outputs
        logs = {'loss_val': avg_loss, 'acc_val': avg_acc}
        return {'val_loss': avg_loss, 'log': logs}

    # ---------- Test ----------
    def test_step(self, batch, batch_idx):
        x, y_all = batch
        y = y_all[:, 4]  # get only label

        # # 1 node output
        # y_logits = self.forward(x)
        # y_logits = y_logits.view(y_logits.shape[0])
        # y = y.type_as(y_logits)
        # acc = self.binary_acc(y_logits, y)

        # 2 node output
        y_logits = self.forward(x)
        y_hat = torch.round(F.softmax(y_logits, dim=-1))
        acc = self.binary_acc(y_hat, y)

        # Specificy the subject from where this sample comes from:
        subj = y_all[:, 0].tolist()
        subj_str = 'acc_' + str(subj[0])

        # # Calculate predicted outputs
        # y_predicted = torch.sigmoid(y_logits)
        # # define 0.5 as threshold
        # y_predicted[y_predicted>=0.5] = 1.0
        # y_predicted[y_predicted<0.5] = 0.0

        # Define outputs
        output = {
            'acc': acc.clone().detach(),
            subj_str: acc.clone().detach(),
            'y_true': y.clone().detach(),
            # 'y_predicted': y_predicted.clone().detach()    # convert list of pairs to y_prediction
            'y_predicted': torch.tensor([int(x[0] == 0) for x in y_hat])  # convert list of pairs to y_prediction
        }
        return output

    def test_epoch_end(self, outputs):

        test_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.test_y_true = torch.stack([x['y_true'] for x in outputs])
        self.test_y_predicted = torch.stack([x['y_predicted'] for x in outputs])

        # Get accuracy per subject
        self.test_acc_subj = [0] * 6
        for i in range(6):
            subj_str = 'acc_' + str(i + 1)
            accuracies = [x[subj_str] for x in outputs if (subj_str in x)]
            if (not accuracies):  # if empty
                self.test_acc_subj[i] = 0
            else:
                self.test_acc_subj[i] = torch.stack(accuracies).mean()

        # Define outputs
        # General accuracy
        logs = {'acc_test': test_acc}
        # Accuracy per subject
        for i in range(6):
            subj_str = 'acc_subj_' + str(i + 1)
            logs[subj_str] = self.test_acc_subj[i]

        # for i in range()
        return {'log': logs}

    # # +++++++++++++++++++++++++++ Get data and split (test, val, train) +++++++++++++++++++++++++++

    # def prepare_data(self):
    #   # All datasets are downloaded passed to the Model Constructor.
    #   # No further preparation needed.
    #   pass

    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # +++++++++++++++++++++++++++ Data Loaders +++++++++++++++++++++++++++
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # def train_dataloader(self):
    #   return DataLoader(self.train_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=True)

    # def val_dataloader(self):
    #   return DataLoader(self.val_dataset, batch_size=self.hyper_params['batch_size'], num_workers=8, shuffle=False)

    # def test_dataloader(self):
    #   return DataLoader(self.test_dataset, batch_size=self.hyper_params['test_batch_size'], num_workers=8)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++ Oprimizer and Scheduler +++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def configure_optimizers(self):
        # Use Stochastic Gradient Discent
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hyper_params['learning_rate'],
                                    momentum=self.hyper_params['momentum'],
                                    weight_decay=self.hyper_params['weight_decay'])

        # scheduler = StepLR(optimizer, step_size=1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # return [optimizer], [scheduler]
        return [optimizer]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++-+++++++++++++++++++++++++++ Auxiliar functions +++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Measure binary accuracy
    # def binary_acc(self, y_logits, y):
    def binary_acc(self, y_hot, y):

        # 1 output node
        # # apply sigmoid to output result (from single node)
        # y_sigmoid = torch.sigmoid(y_logits)
        # # define 0.5 as threshold
        # y_sigmoid[y_sigmoid>=0.5] = 1
        # y_sigmoid[y_sigmoid<0.5] = 0
        # # Calculate accuracy (sum of all inputs equal to the targets divided by total number of targets)
        # acc = 100 * (y_sigmoid == y).sum().type(torch.float) / len(y)

        # 2 output node
        correct = torch.tensor([y_hot[idx, y[idx]] for idx in range(len(y))])

        correct_results_sum = correct.sum()
        acc = correct_results_sum / len(y)
        acc = torch.round(acc * 100)

        return acc

    # Return test y_true and y_predicted vectors for Confusion Matrix in Comet ML
    def get_test_labels_predictions(self):
        return (self.test_y_true, self.test_y_predicted)
