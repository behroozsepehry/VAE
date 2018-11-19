from torch import nn


class VaeModelBase(nn.Module):
    def __init__(self):
        super(VaeModelBase, self).__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
