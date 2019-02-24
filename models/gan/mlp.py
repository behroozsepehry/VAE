from models.gan import base
from models import mlp


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        g_layer_sizes = kwargs.get('g_layer_sizes')
        d_layer_sizes = kwargs.get('d_layer_sizes')
        g_activation = kwargs.get('g_activation')
        ngpu = kwargs.get('ngpu', 1)

        assert g_layer_sizes[0] == self.z_args['z_dim']
        assert d_layer_sizes[-1] == 1
        self.discriminator = mlp.Model(d_layer_sizes, activation='Sigmoid', ngpu=ngpu)
        self.generator = mlp.Model(g_layer_sizes, activation=g_activation, ngpu=ngpu)
