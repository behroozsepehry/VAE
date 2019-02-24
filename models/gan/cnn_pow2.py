from models.gan import base
from models import conv_pow2, deconv_pow2


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        in_channels = kwargs.get('in_channels', 1)
        in_size = kwargs['in_size']
        self.discriminator = conv_pow2.Model(in_size, in_channels, mid_channels, 1, activation='Sigmoid')
        self.generator = deconv_pow2.Model(in_size, self.z_args['z_dim'], mid_channels, in_channels, activation='Sigmoid')
