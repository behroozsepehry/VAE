import models.base
from utilities import sampler


class GanModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(GanModelBase, self).__init__(*args, **kwargs)
        self.z_dim = kwargs['z_dim']
        self.discriminator = None
        self.generator = None
        self.z_generator = sampler.Sampler(**kwargs['z_args'])

    def generate(self, device, *args, **kwargs):
        n_samples = kwargs.get('n_samples', 1)
        z = self.z_generator((n_samples, self.z_dim)).to(device)
        x = self.generator(z)
        return x, z

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x_real, = tuple(args)
        x_fake, z_fake = self.generate(x_real.device, n_samples=x_real.size(0))
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        return x_real, y_real, x_fake, y_fake
