from models.gan import base
from models import conv_32, deconv_32


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        self.discriminator = conv_32.Model(1, mid_channels, 1, activation='Sigmoid')
        self.generator = deconv_32.Model(self.z_args['z_dim'], mid_channels, 3, activation='Sigmoid')
