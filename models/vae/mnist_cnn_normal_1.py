import torch
from torch import nn

from models.vae import normal_base
from models import conv_28, deconv_28


class Model(normal_base.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        z_dim = self.z_args['z_dim']
        self.encoder = conv_28.Model(1, mid_channels, 2 * z_dim)
        self.decoder = deconv_28.Model(z_dim, mid_channels, 2)

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
