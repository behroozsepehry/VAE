import torch
from torch import nn

from models.vae import base_normal
from models import conv_28, deconv_28


class Model(base_normal.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.encoder = None
        self.decoder = None

    def encode(self, x, **kwargs):
        zz = self.encoder(x)
        z_mu, z_logvar = torch.split(zz, zz.size(1)//2, dim=1)
        return dict(z_mu=z_mu, z_logvar=z_logvar)

    def _decode(self, z, **kwargs):
        xx = self.decoder(z)
        x_mu, x_logvar = torch.split(xx, xx.size(1)//2, dim=1)
        x_mu = torch.sigmoid(x_mu)
        x_logvar = torch.clamp(x_logvar, min=-10.)
        return dict(x_mu=x_mu, x_logvar=x_logvar)
