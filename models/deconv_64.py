from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(Model, self).__init__()
        activation_name = kwargs.get('activation')
        activation = []
        if activation_name:
            activation.append(getattr(nn, activation_name)())
        self.cnn = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     in_channels, mid_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels * 8),
            nn.ReLU(True),
            # state size. (mid_channels*8) x 4 x 4
            nn.ConvTranspose2d(mid_channels * 8, mid_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels * 4),
            nn.ReLU(True),
            # state size. (mid_channels*4) x 8 x 8
            nn.ConvTranspose2d(mid_channels * 4, mid_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels * 2),
            nn.ReLU(True),
            # state size. (mid_channels*2) x 16 x 16
            nn.ConvTranspose2d(mid_channels * 2,     mid_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            # state size. (mid_channels) x 32 x 32
            nn.ConvTranspose2d(    mid_channels,      out_channels, 4, 2, 1, bias=False),
            # state size. (out_channels) x 64 x 64
            *activation
        )

    def forward(self, x):
        return self.cnn(x.view(x.size(0), -1, 1, 1))


if __name__ == '__main__':
    import torch
    model = Model(20, 10, 1, activation='Sigmoid')
    z = torch.randn(2, 20)
    x = model(z)
    print(x)
    print(x.size())
