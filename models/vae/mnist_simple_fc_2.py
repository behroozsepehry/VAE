import torch
from torch import nn
from torch.nn import functional as F

from models.vae import mnist_simple_fc_normal_1, base


class Model(base.VaeModelBase):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        super(Model, self).__init__(*args, **kwargs)
        self.z_dim = kwargs['z_dim']
        self.z2_dim = kwargs['z2_dim']
        self.model_1 = mnist_simple_fc_normal_1.Model(**kwargs)

        self.fc1 = nn.Linear(self.z_dim, self.z2_dim)
        self.fc2 = nn.Linear(self.z2_dim, self.z_dim)

    def encode(self, *args, **kwargs):
        return self.model_1.encode(*args, **kwargs)

    def reparameterize(self, *args, **kwargs):
        return self.model_1.reparameterize(*args, **kwargs)

    def decode(self, z, **kwargs):
        if self.training:
            z2 = torch.sigmoid(self.fc1(z))
        else:
            z2 = (torch.sign(self.fc1(z)) + 1.)/2.
        h2 = F.relu(self.fc2(z2))
        return self.model_1.decode(h2) + (z2,)

    def forward(self, x, **kwargs):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon, z2 = self.decode(z)
        return x_recon, mu, logvar, z2
