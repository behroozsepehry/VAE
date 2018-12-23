import torch
from torch import nn
from torch.nn import functional as F

from models import base


class VaeModelNormalBase(base.VaeModelBase):
    def __init__(self, **kwargs):
        super(VaeModelNormalBase, self).__init__()
        z_dim = kwargs['z_dim']
        self.z_dim = z_dim
        self.sampling_iterations = kwargs.get('sampling_iterations', 0)

    def reparameterize(self, *args, **kwargs):
        assert len(args) == 2
        mu, logvar = tuple(args)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def sample(self, device, *args, **kwargs):
        n_samples = kwargs.get('n_samples', 1)
        z = torch.randn((n_samples, self.z_dim)).to(device)
        x_mu, _ = self.decode(z)
        sampling_iterations = kwargs.get('sampling_iterations', self.sampling_iterations)
        assert hasattr(sampling_iterations, '__len__')
        max_sampling_iterations = max(sampling_iterations)
        x_mu_list = []
        for i in range(0, max_sampling_iterations+1):
            if i in sampling_iterations:
                x_mu_list.append(x_mu)
            z_mu, _ = self.encode(x_mu)
            x_mu, _ = self.decode(z_mu)
        return x_mu_list
