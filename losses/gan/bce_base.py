import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class BceLossBase(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(BceLossBase, self).__init__(*args, **kwargs)
        self.real_label = kwargs.get('real_label', 1)
        self.fake_label = kwargs.get('fake_label', 0)

    def _check_args(self, *args, **kwargs):
        assert len(args) == 2
        assert len(args[0]) == 2
        assert len(args[1]) == 2

    def __call__(self, *args, **kwargs):
        return NotImplementedError
