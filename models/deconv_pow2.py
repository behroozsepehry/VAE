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
        for i in range(1, n_mid_layers+1).__reversed__():
            mid_layers += [nn.ConvTranspose2d(mid_channels * 2**i, mid_channels * 2**(i-1), 4, 2, 1, bias=False),
                           nn.BatchNorm2d(mid_channels * 2**(i-1)),
                           nn.ReLU(True),]

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels * 2**n_mid_layers, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels * 2**n_mid_layers),
            nn.ReLU(True),
            *mid_layers,
            nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, 1, bias=False),
            *activation
        )

    def forward(self, x):
        return self.cnn(x.view(x.size(0), -1, 1, 1))


if __name__ == '__main__':
    import torch
    model = Model(128, 20, 10, 1, activation='Sigmoid')
    z = torch.randn(2, 20)
    x = model(z)
    print(x)
    print(x.size())
