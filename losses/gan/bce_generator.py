import torch
import torch.nn
from torch.nn import functional as F

from losses.gan import bce_base


class Loss(bce_base.BceLossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def _create_label_tensors(self, y_fake):
        l_fake = torch.full(y_fake.size(), self.FAKE_LABEL, device=y_fake.device)
        return l_fake

    def _compute_losses(self, y_fake, l_fake):
        bce_fake = F.binary_cross_entropy(y_fake, l_fake, reduction='sum')
        return bce_fake

    def __call__(self, *args, **kwargs):
        self._check_args(*args, **kwargs)
        _, _, _, y_fake = tuple(args)
        l_fake = self._create_label_tensors(y_fake)
        bce_fake = self._compute_losses(y_fake, l_fake)
        loss_generator = -bce_fake
        return loss_generator
