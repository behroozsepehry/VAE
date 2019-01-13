from losses import base
from losses.vae import mse


class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()
        self.mse_loss = mse.Loss(*args, **kwargs)
        self.l1_weight = kwargs.get('l1_weight', 1.0)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_mu, x_logvar, z_2, z_mu, z_logvar, **kwargs):
        vae_1_loss_val = self.mse_loss(x, x_mu, x_logvar, z_mu, z_logvar)
        l1 = z_2.abs().sum()/z_2.size(1)
        loss_val = vae_1_loss_val + self.l1_weight * l1
        return loss_val
