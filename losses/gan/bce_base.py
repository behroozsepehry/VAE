import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class BceLossBase(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(BceLossBase, self).__init__()
        self.REAL_LABEL = kwargs.get('REAL_LABEL', 1)
        self.FAKE_LABEL = kwargs.get('FAKE_LABEL', 0)

    def __call__(self, *args, **kwargs):
        return NotImplementedError
