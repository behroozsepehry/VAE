from models.gan import base
from models import conv_28, deconv_28


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        self.discriminator = conv_28.Model(1, mid_channels, 1, activation='Sigmoid')
        self.generator = deconv_28.Model(self.z_args['z_dim'], mid_channels, 1, activation='Sigmoid')
