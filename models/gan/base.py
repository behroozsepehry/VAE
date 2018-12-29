import torch
from torch import nn

import models.base


class GanModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(GanModelBase, self).__init__(*args, **kwargs)

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def discriminate(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
