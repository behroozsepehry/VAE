import torch
import torch.nn
from torch.nn import functional as F

from losses.gan import bce_base


class Loss(bce_base.BceLossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def _create_label_tensors(self, y_real, y_fake):
        l_real = torch.full(y_real.size(), self.REAL_LABEL, device=y_real.device)
        l_fake = torch.full(y_fake.size(), self.FAKE_LABEL, device=y_fake.device)
        return l_real, l_fake

    def _compute_losses(self, y_real, l_real, y_fake, l_fake):
        bce_real = F.binary_cross_entropy(y_real, l_real, reduction='sum')
        bce_fake = F.binary_cross_entropy(y_fake, l_fake, reduction='sum')
        return bce_real, bce_fake

    def __call__(self, *args, **kwargs):
        self._check_args(*args, **kwargs)
        x_real, y_real, x_fake, y_fake = tuple(args)
        l_real, l_fake = self._create_label_tensors(y_real, y_fake)
        bce_real, bce_fake = self._compute_losses(y_real, l_real, y_fake, l_fake)
        loss_discriminator = bce_real + bce_fake
        return loss_discriminator
