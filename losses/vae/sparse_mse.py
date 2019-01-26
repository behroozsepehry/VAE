import torch

from losses import base
from losses.vae import mse


class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()
        self.mse_loss = mse.Loss(*args, **kwargs)
        self.l1_weight = kwargs.get('l1_weight', 1.0)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_mu, x_logvar, z_2, z_mu, z_logvar, **kwargs):
        tol = kwargs.get('tol', 1.0e-8)
        vae_1_loss_vals = self.mse_loss(x, x_mu, x_logvar, z_mu, z_logvar)
        l1 = z_2.abs().sum()/z_2.size(1)
        l1_loss = self.l1_weight * l1
        nz = torch.sum(z_2 < tol)
        loss_vals = dict(**vae_1_loss_vals, l1=l1_loss, nz=nz)
        loss_vals['mse_kld'] = loss_vals['loss']
        loss_vals['loss'] += l1_loss
        return loss_vals
