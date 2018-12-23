import torch
from torch import nn
from torch.nn import functional as F

from models import normal_base


class Model(normal_base.VaeModelNormalBase):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 16, 5, 5
            nn.Conv2d(16, 32, 3, stride=1, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
            nn.Conv2d(32, 32, 3, stride=1, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), # b, 8, 2, 2
            nn.Conv2d(32, 16, 3, stride=1, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b
        )

        self.fc11 = nn.Linear(16 * 16 * 16, self.z_dim)
        self.fc12 = nn.Linear(16 * 16 * 16, self.z_dim)

        self.fc2 = nn.Sequential(nn.Linear(self.z_dim, 7 * 7 * 32),
                                 nn.ReLU(True),
                                 )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=1),  # b, 8, 15, 15
            nn.ReLU(True),
        )
        self.deconv_mu = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, stride=2, padding=2),  # b, 1, 28, 28
            nn.Sigmoid(),
        )
        self.deconv_logvar = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, stride=2, padding=2),  # b, 1, 28, 28
        )

    def encode(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        h1 = self.conv(x).view(x.size(0), -1)
        z_mu, z_logvar = self.fc11(h1), self.fc12(h1)
        return z_mu, z_logvar

    def decode(self, *args, **kwargs):
        assert len(args) == 1
        z, = tuple(args)
        h2 = self.fc2(z).view(z.size(0), 32, 7, 7)
        h3 = self.deconv(h2)
        x_mu, x_logvar = self.deconv_mu(h3), torch.clamp(self.deconv_logvar(h3), min=-10.)
        return x_mu, x_logvar
