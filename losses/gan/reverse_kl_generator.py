import torch
import torch.nn
from torch.nn import functional as F

from losses.gan import bce_base


class Loss(bce_base.BceLossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def _compute_losses(self, y_fake):
        r_kl = torch.sum(torch.log(1./y_fake - 1.))
        return r_kl

    def __call__(self, x_real, y_real, x_fake, y_fake, **kwargs):
        r_kl = self._compute_losses(y_fake)
        return dict(loss=r_kl)
