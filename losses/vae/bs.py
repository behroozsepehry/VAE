import torch
import torch.nn
from torch.nn import functional as F

from losses import base
from utilities import nn_utilities

class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_mu, z_mu, z_logvar, **kwargs):
        eps = kwargs.get('eps', 1.0e-7)
        bs = -torch.sum(torch.log(nn_utilities.BsDist.pdf(x, x_mu) + eps))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        return dict(loss=bs+kld, bs=bs, kld=kld)
