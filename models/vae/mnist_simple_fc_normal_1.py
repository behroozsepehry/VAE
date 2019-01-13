import torch
from torch import nn
from torch.nn import functional as F

from models.vae import normal_base


class Model(normal_base.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        z_dim = self.z_args['z_dim']

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, z_dim)
        self.fc22 = nn.Linear(400, z_dim)
        self.fc3 = nn.Linear(z_dim, 400)
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

    def encode(self, x, **kwargs):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z_mu, z_logvar = self.fc21(h1), self.fc22(h1)
        return dict(z_mu=z_mu, z_logvar=z_logvar)

    def _decode(self, z, **kwargs):
        h3 = F.relu(self.fc3(z))
        x_mu, x_logvar = torch.sigmoid(self.fc41(h3)), torch.clamp((self.fc42(h3)), min=-10.)
        return dict(x_mu=x_mu, x_logvar=x_logvar)
