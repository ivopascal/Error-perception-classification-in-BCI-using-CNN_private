import math

import torch
import torch.nn as nn
from torch import Tensor


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


class Squeeze(nn.Module):
    # torch.Tensor.Squeeze() implemented as a NN layer
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f'Squeeze'

    def forward(self, x: Tensor):
        print(x.shape)
        out = x.squeeze()
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


class SamplingSoftmax(nn.Module):
    """
        Sigmoid activation with Gaussian logits. Receives mean/variance logits and computes the softmax output through sampling.
    """

    def __init__(self, num_samples=50, temperature=1.0, variance_type="linear_std") -> None:
        """
        Args:
            num_samples: Number of samples used to compute the softmax approximation.
                         This parameter controls the trade-off between computation and approximation quality.
            variance_type: Assumptions made on the variance input, possible values are:
                logit: Input is a variance logit, an exponential transformation will be applied to produce standard deviation.
                linear_std: Input is standard deviation, no transformations are applied.
                linear_variance: Input is variance, square root will be applied to obtain standard deviation.
        """
        super().__init__()

        assert variance_type in ["logit", "linear_std", "linear_variance"]

        self.num_samples = num_samples
        self.temperature = temperature
        self.variance_type = variance_type
        self.samples = None

    def __repr__(self):
        return f'SamplingSoftmax'

    def preprocess_variance_input(self, var_input):
        if self.variance_type == "logit":
            return torch.exp(var_input)

        if self.variance_type == "linear_variance":
            return torch.sqrt(var_input)

        return var_input

    def forward(self, inputs):
        assert len(inputs) == 2, "This layer requires exactly two inputs (mean and variance logits)"

        logit_mean, logit_var = inputs
        logit_std = self.preprocess_variance_input(logit_var)
        repetitions = [1, self.num_samples, 1]

        logit_mean = torch.unsqueeze(logit_mean, dim=1)
        logit_mean = logit_mean.repeat(repetitions)

        logit_std = torch.unsqueeze(logit_std, dim=1)
        logit_std = logit_std.repeat(repetitions)

        logit_samples = torch.randn_like(logit_mean) * logit_std + logit_mean

        # Apply max normalization for numerical stability
        logit_samples = logit_samples - torch.max(logit_samples, dim=-1, keepdim=True)[0]

        # Apply temperature scaling to logits
        logit_samples = logit_samples / self.temperature

        prob_samples = torch.softmax(logit_samples, dim=-1)
        self.samples = prob_samples
        probs = torch.mean(prob_samples, dim=1)

        probs_variance = torch.var(prob_samples, dim=1)

        # This is required due to approximation error, without it probabilities can sum to 1.01 or 0.99
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)

        return probs, probs_variance


class Softplus(nn.Module):

    def __init__(self, threshold=20):
        super().__init__()
        self.threshold = threshold

    def __repr__(self):
        return "Softplus"

    def forward(self, x):
        out = torch.log(torch.exp(x) + 1)
        out[x > self.threshold] = x[x > self.threshold]  # Linear for numerical stability

        return out
