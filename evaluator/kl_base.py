import torch

from utilities import general_utilities as g_util
from utilities import main_utilities as m_util
from utilities import nn_utilities as nn_util
from losses import base as loss_base
from losses.gan import bce_discriminator
from losses.gan import reverse_kl_generator
from models.gan import base as gan_base


class GanFixedG(gan_base.GanModelBase):
    def __init__(self, generator, discriminator, **kwargs):
        super(GanFixedG, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator

    def generate(self, device, **kwargs):
        return self.generator.generate(device, **kwargs)

    def forward_backward(self, x, loss_functions, optimizers, **kwargs):
        d_optim = optimizers['discriminator']
        d_loss_func = loss_functions['discriminator']

        x_real = x
        y_real = self.discriminator(x_real)
        x_fake = self.generate(x_real.device, n_samples=x_real.size(0))['x']

        y_fake = self.discriminator(x_fake.detach())
        d_optim.zero_grad()
        d_loss = d_loss_func(x_real, y_real, x_fake.detach(), y_fake)
        d_loss['loss'].backward()
        d_optim.step()

        losses = dict(**g_util.append_key_dict(d_loss, 'discriminator_'))
        return dict(losses=losses)


class Evaluator(object):
    def __init__(self, generator, dataloaders, device, n_samples, discriminator_args, optim_args, **kwargs):
        self.generator = generator
        self.discriminator = m_util.get_model(**discriminator_args)
        self.discriminator.to(device)
        self.dataloaders = dataloaders
        self.optimizer = m_util.get_optimizer(self.discriminator.parameters(), **optim_args)
        self.trained = False
        self.device = device
        self.n_samples = n_samples
        self.gan_fixed_g = GanFixedG(self.generator, self.discriminator, **kwargs)
        self.gan_fixed_g.to(self.device)

    def train(self):
        if not self.trained:
            self.gan_fixed_g.train_model(self.device, self.dataloaders,
                                    dict(discriminator=self.optimizer),
                                    dict(discriminator=bce_discriminator.Loss()), None)
        self.trained = True

    def __call__(self):
        raise NotImplementedError
