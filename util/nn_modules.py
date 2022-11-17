import torch.nn as nn


class View(nn.Module):
    # torch.Tensor.view() implemented as a NN layer
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, x):
        print(x.shape)
        batch_size = x.size(0)
        shape = (batch_size, *self.shape)
        out = x.view(shape)
        print(out.shape)
        return out
