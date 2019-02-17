from utilities import general_utilities as g_util
from utilities import main_utilities as m_util
from utilities import nn_utilities as nn_util
from losses import base as loss_base
from losses.gan import bce_discriminator
from losses.gan import reverse_kl_generator
from models.gan import base as gan_base


class GanFixedG(gan_base.GanModelBase):
    def __init__(self, generator, discriminator, optim_args):
        super(GanFixedG, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.loss_func = bce_discriminator.Loss()
        self.optim = m_util.get_optimizer(self.discriminator.parameters(), **optim_args)

    def generate(self, device, **kwargs):
        return self.generator.generate(device, **kwargs)

    def forward_backward(self, x, loss_functions, optimizers, **kwargs):
        x_real = x
        y_real = self.discriminator(x_real)
        x_fake = self.generate(x_real.device, n_samples=x_real.size(0))['x']

        y_fake = self.discriminator(x_fake.detach())
        self.optim.zero_grad()
        d_loss = self.loss_func(x_real, y_real, x_fake.detach(), y_fake)
        d_loss['loss'].backward()
        self.optim.step()

        losses = dict(**g_util.append_key_dict(d_loss, 'discriminator_'))
        return dict(losses=losses)


class Loss(loss_base.LossBase):
    def __init__(self, generator, discriminator, optim_args):
        super(Loss, self).__init__()
        self.gan_fixed_g = GanFixedG(generator, discriminator, optim_args)

    def __call__(self, device, dataloaders, optimizers, losses, logger, **kwargs):
        self.gan_fixed_g.train_model(device, dataloaders, optimizers, losses, logger, **kwargs)
        result = nn_util.apply_func_to_model_data(self.gan_fixed_g, reverse_kl_generator.Loss(), dataloaders['train'], device)
        return result
