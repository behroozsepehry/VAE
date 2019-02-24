from models.vae import base_nn_normal
from models import conv_28, deconv_28


class Model(base_nn_normal.Model):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        in_channels = kwargs.get('in_channels', 1)
        z_dim = self.z_args['z_dim']
        ngpu = kwargs.get('ngpu', 1)

        self.encoder = conv_28.Model(in_channels, mid_channels, 2 * z_dim, ngpu=ngpu)
        self.decoder = deconv_28.Model(z_dim, mid_channels, 2 * in_channels, ngpu=ngpu)
