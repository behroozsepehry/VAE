import torch
import torch.nn
from torch.nn import functional as F

from losses.gan import bce_generator


class Loss(bce_generator.Loss):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def _create_label_tensors(self, y_fake):
        l_fake = torch.full(y_fake.size(), self.REAL_LABEL, device=y_fake.device)
        return l_fake

    def __call__(self, x_real, y_real, x_fake, y_fake, **kwargs):
        loss_generator = super(Loss, self).__call__(x_real, y_real, x_fake, y_fake, **kwargs)
        loss_generator['loss'] = -loss_generator['loss']
        return loss_generator
