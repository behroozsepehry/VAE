from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(Model, self).__init__()
        activation_name = kwargs.get('activation')
        activation = []
        if activation_name:
            activation.append(getattr(nn, activation_name)())
        self.cnn = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, mid_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid_channels) x 32 x 32
            nn.Conv2d(mid_channels, mid_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid_channels*2) x 16 x 16
            nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid_channels*4) x 8 x 8
            nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid_channels*8) x 4 x 4
            nn.Conv2d(mid_channels * 8, out_channels, 4, 1, 0, bias=False),
            *activation
        )

    def forward(self, x):
        return self.cnn(x).view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    model = Model(1, 10, 20, activation='Sigmoid')
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print(y)
    print(y.size())
