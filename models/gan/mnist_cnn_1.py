from models.gan import base
from models import cnn_conv_1, cnn_deconv_1


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        mid_channels = kwargs.get('mid_channels', 1)
        self.discriminator = cnn_conv_1.Model(1, mid_channels, 1, activation='Sigmoid')
        self.generator = cnn_deconv_1.Model(self.z_args['z_dim'], mid_channels, 1, activation='Sigmoid')
