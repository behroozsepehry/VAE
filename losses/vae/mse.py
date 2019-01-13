import numpy as np
import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_mu, x_logvar, z_mu, z_logvar, **kwargs):
        mse = 0.5 * (torch.sum(x_logvar.exp() + (((x_mu - x.view(x_mu.size())).pow(2))/(x_logvar.exp())))
                     + z_mu.size(0) * z_mu.size(1) * np.log(2*np.pi))
        # print(x_logvar.min(), x_logvar.mean())
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        return mse + kld
