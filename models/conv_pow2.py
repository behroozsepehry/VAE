import numpy as np
from torch import nn

from utilities import general_utilities as g_util


class Model(nn.Module):
    def __init__(self, in_size, in_channels, mid_channels, out_channels, **kwargs):
        super(Model, self).__init__()
        assert g_util.is_power2(in_size)
        n_mid_layers = int(np.log2(in_size)) - 3

        activation_name = kwargs.get('activation')
        activation = []
        if activation_name:
            activation.append(getattr(nn, activation_name)())

        mid_layers = []
        for i in range(1, n_mid_layers+1):
            mid_layers += [nn.Conv2d(mid_channels * 2**(i-1), mid_channels * 2**i, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(mid_channels * 2**i),
                           nn.LeakyReLU(0.2, inplace=True),]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            *mid_layers,
            nn.Conv2d(mid_channels * 2**n_mid_layers, out_channels, 4, 1, 0, bias=False),
            *activation
        )

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    model = Model(128, 1, 10, 20, activation='Sigmoid')
    print(model)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(y)
    print(y.size())
