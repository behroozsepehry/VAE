import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_mu, z_mu, z_logvar, **kwargs):
        bce = F.binary_cross_entropy(x_mu, x.view(x_mu.size()), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        return dict(loss=bce+kld, bce=bce, kld=kld)
