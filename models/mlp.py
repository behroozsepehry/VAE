from torch import nn


class Model(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):