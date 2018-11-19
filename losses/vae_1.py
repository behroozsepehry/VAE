import torch
from torch.nn import functional as F

from losses import base


class Loss(base.VaeLossBase):

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, *args, **kwargs):
        if len(args) != 4:
            raise AssertionError('the function needs 4 arguments')
        x, recon_x, mu, logvar = tuple(args)
        BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.size()), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD