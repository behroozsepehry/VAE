from models.gan import base
from models import mnist_conv_1, mnist_deconv_1


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        self.discriminator = mnist_conv_1.Model(1, mid_channels, 1, activation='Sigmoid')
        self.generator = mnist_deconv_1.Model(self.z_args['z_dim'], mid_channels, 1, activation='Sigmoid')
