import torch
from torch import nn
from torch.nn import functional as F

from models.vae import normal_base


class Model(normal_base.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.z_dim)
        self.fc22 = nn.Linear(400, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 400)
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

    def encode(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z_mu, z_logvar = self.fc21(h1), self.fc22(h1)
        return z_mu, z_logvar

    def decode(self, *args, **kwargs):
        assert len(args) == 1
        z, = tuple(args)
        h3 = F.relu(self.fc3(z))
        x_mu, x_logvar = torch.sigmoid(self.fc41(h3)), torch.clamp((self.fc42(h3)), min=-10.)
        return x_mu, x_logvar
