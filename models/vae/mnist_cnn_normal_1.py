import torch
from torch import nn

from models.vae import normal_base


class Model(normal_base.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 16, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )

        z_dim = self.z_args['z_dim']
        self.fc11 = nn.Linear(16 * 13 * 13, z_dim)
        self.fc12 = nn.Linear(16 * 13 * 13, z_dim)

        self.fc2 = nn.Sequential(nn.Linear(z_dim, 7 * 7 * 32),
                                 nn.ReLU(True),
                                 )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=1),
            nn.ReLU(True),
        )
        self.deconv_mu = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, stride=2, padding=2),
            nn.Sigmoid(),
        )
        self.deconv_logvar = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, stride=2, padding=2),
        )

    def encode(self, x, **kwargs):
        h1 = self.conv(x).view(x.size(0), -1)
        z_mu, z_logvar = self.fc11(h1), self.fc12(h1)
        return dict(z_mu=z_mu, z_logvar=z_logvar)

    def _decode(self, z, **kwargs):
        h2 = self.fc2(z).view(z.size(0), 32, 7, 7)
        h3 = self.deconv(h2)
        x_mu, x_logvar = self.deconv_mu(h3), torch.clamp(self.deconv_logvar(h3), min=-10.)
        return dict(x_mu=x_mu, x_logvar=x_logvar)
