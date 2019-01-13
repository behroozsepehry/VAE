import numpy as np

import models.base
from utilities import sampler


class GanModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(GanModelBase, self).__init__(*args, **kwargs)
        self.z_args = kwargs['z_args']
        self.discriminator = None
        self.generator = None
        self.z_generator = sampler.Sampler(**kwargs['z_args'])

    def generate(self, device, **kwargs):
        n_samples = kwargs.get('n_samples', 1)
        z = self.z_generator((n_samples, self.z_args['z_dim'])).to(device)
        x = self.generator(z)
        return dict(x=x, z=z)

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x_real, = tuple(args)
        x_fake, z_fake = self.generate(x_real.device, n_samples=x_real.size(0))
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)
        return dict(x_real=x_real, y_real=y_real, x_fake=x_fake, y_fake=y_fake)

    def forward_backward(self, *args, **kwargs):
        assert len(args) == 3
        x_real, (d_loss_func, g_loss_func), (d_optim, g_optim) = args
        y_real = self.discriminator(x_real)
        x_fake, z_fake = self.generate(x_real.device, n_samples=x_real.size(0))

        y_fake = self.discriminator(x_fake.detach())
        d_optim.zero_grad()
        d_loss = d_loss_func(x_real, y_real, x_fake.detach(), y_fake)
        d_loss.backward()
        d_optim.step()

        y_fake = self.discriminator(x_fake)
        g_optim.zero_grad()
        g_loss = g_loss_func(None, None, x_fake, y_fake)
        g_loss.backward()
        g_optim.step()

        losses = np.array([d_loss.item(), g_loss.item()])
        # print(y_real.mean().item(), y_fake.mean().item())
        # print(d_optim.param_groups[0]['params'][0].grad.cpu().numpy())
        # print('######################')
        # print(g_optim.param_groups[0]['params'][0].grad.cpu().numpy())

        return dict(losses=losses)

    def get_parameters_groups(self):
        return [self.discriminator.parameters(), self.generator.parameters()]
