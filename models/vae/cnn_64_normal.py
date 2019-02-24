from models.vae import base_nn_normal
from models import conv_64, deconv_64


class Model(base_nn_normal.Model):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        in_channels = kwargs.get('in_channels', 1)
        z_dim = self.z_args['z_dim']

        self.encoder = conv_64.Model(in_channels, mid_channels, 2 * z_dim)
        self.decoder = deconv_64.Model(z_dim, mid_channels, 2 * in_channels)
