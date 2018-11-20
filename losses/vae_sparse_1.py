import torch
from torch.nn import functional as F

from losses import base, vae_1


class Loss(base.VaeLossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()
        self.vae1_loss = vae_1.Loss(*args, **kwargs)
        self.l1_weight = kwargs.get('l1_weight', 1.0)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, *args, **kwargs):
        assert len(args) == 5
        x, recon_x, mu, logvar, z2 = tuple(args)
        vae_1_loss_val = self.vae1_loss(x, recon_x, mu, logvar)
        l1 = z2.abs().sum()/z2.size(1)
        loss_val = vae_1_loss_val + self.l1_weight * l1
        return loss_val