from models.gan import base
from models import conv_28, deconv_28


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        in_channels = kwargs.get('in_channels', 1)
        ngpu = kwargs.get('ngpu', 1)

        self.discriminator = conv_28.Model(in_channels, mid_channels, 1, activation='Sigmoid', ngpu=ngpu)
        self.generator = deconv_28.Model(self.z_args['z_dim'], mid_channels, in_channels, activation='Sigmoid', ngpu=ngpu)
