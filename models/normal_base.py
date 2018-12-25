import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from models import base


class VaeModelNormalBase(base.VaeModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelNormalBase, self).__init__(*args, **kwargs)
        z_dim = kwargs['z_dim']
        self.z_dim = z_dim
        self.sampling_iterations = kwargs.get('sampling_iterations', [0])

    def reparameterize(self, *args, **kwargs):
        assert len(args) == 2
        mu, logvar = tuple(args)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def sample(self, device, *args, **kwargs):
        n_samples = kwargs.get('n_samples', 1)
        z_mu = torch.randn((n_samples, self.z_dim)).to(device)
        return self.do_sampling_iterations(z_mu, device, *args, **kwargs)

    def do_sampling_iterations(self, z_mu, device, *args, **kwargs):
        sampling_iterations = kwargs.get('sampling_iterations', self.sampling_iterations)
        assert hasattr(sampling_iterations, '__len__')
        max_sampling_iterations = max(sampling_iterations)

        x_mu, _ = self.decode(z_mu)

        x_mu_list = []
        z_mu_list = []
        for i in range(0, max_sampling_iterations+1):
            if i in sampling_iterations:
                x_mu_list.append(x_mu)
                z_mu_list.append(z_mu)
            z_mu, _ = self.encode(x_mu)
            x_mu, _ = self.decode(z_mu)

        return x_mu_list, z_mu_list

    def get_sampling_iterations_loss(self, dataloader, loss_function, device, *args, **kwargs):
        sampling_iterations = kwargs.get('sampling_iterations', self.sampling_iterations)
        assert hasattr(sampling_iterations, '__len__')
        max_sampling_iterations = max(sampling_iterations)
        loss_sampling_iterations = range(max_sampling_iterations)
        self.eval()

        losses = np.zeros(len(loss_sampling_iterations))
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                z_params = self.encode(data)
                z = self.reparameterize(*z_params)
                x_mu_list, z_mu_list = self.do_sampling_iterations(z,
                                                                   device,
                                                                   sampling_iterations=loss_sampling_iterations)
                for i, iteration in enumerate(loss_sampling_iterations):
                    x_params = self.decode(z_mu_list[i])
                    loss = loss_function(*((data,)+x_params+z_params))
                    losses[i] += loss.item()

        losses /= len(dataloader.dataset)
        return losses
