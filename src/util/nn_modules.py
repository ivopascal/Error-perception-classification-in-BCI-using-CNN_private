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


class Permute(nn.Module):
    # torch.Tensor.view() implemented as a NN layer
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def __repr__(self):
        return f'Permute{self.permutation}'

    def forward(self, x):
        out = x.permute(*self.permutation)
        return out


def enable_dropout(model: nn.Module):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class DepthwiseConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 depth_multiplier=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros'
                 ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
