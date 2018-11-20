from torch import nn


class VaeModelBase(nn.Module):
    def __init__(self):
        super(VaeModelBase, self).__init__()

    def encode(self, *args, **kwargs):
        # return tuple
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        # return tuple
        # the first output must be reconstruction
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        # return tuple
        # the first output must be reconstruction
        # second output must be same dimension as z
        raise NotImplementedError
