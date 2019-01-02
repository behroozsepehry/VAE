import torch
import torch.nn
from torch.nn import functional as F

from losses.gan import bce_generator


class Loss(bce_generator.Loss):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__(*args, **kwargs)

    def _create_label_tensors(self, y_fake):
        l_fake = torch.new_full(y_fake.size(), self.real_label, device=y_fake.device)
        return l_fake

    def __call__(self, *args, **kwargs):
        loss_generator = -super(Loss, self).__call__(*args, **kwargs)
        return loss_generator
