import torch
from torch import nn

import models.base


class VaeModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__(*args, **kwargs)

    def encode(self, *args, **kwargs):
        # return tuple
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        # return tuple
        # the first output must be reconstruction
        raise NotImplementedError

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        z_params = self.encode(x)
        z = self.reparameterize(*z_params)
        x_params = self.decode(z)
        return x_params + z_params
