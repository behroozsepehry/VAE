import torch
from torch import nn
from torch.nn import functional as F

from models import base


class Model(base.VaeModelBase):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        z_dim = kwargs['z_dim']
        self.z_dim = z_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, *args, **kwargs):
        assert len(args) == 2
        mu, logvar = tuple(args)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, *args, **kwargs):
        assert len(args) == 1
        z, = tuple(args)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)),

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z) + (mu, logvar)