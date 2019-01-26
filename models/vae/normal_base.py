import time
import torch

from models.vae import base
from utilities import sampler as smp


class VaeModelNormalBase(base.VaeModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelNormalBase, self).__init__(*args, **kwargs)
        self.z_args = kwargs['z_args']
        self.sampling_iterations = kwargs.get('sampling_iterations', [0])
        self.z_generator = smp.Sampler(name='standard_normal')

    def reparameterize(self, z_mu, z_logvar, **kwargs):
        std = torch.exp(0.5*z_logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)
        return dict(z=z)

    def generate(self, device, **kwargs):
        n_samples = kwargs.get('n_samples', 1)
        z = self.z_generator((n_samples, self.z_args['z_dim'])).to(device)
        samples = self.do_sampling_iterations(z, device, **kwargs)
        return samples

    def do_sampling_iterations(self, z_mu, device, **kwargs):
        sampling_iterations = kwargs.get('sampling_iterations', self.sampling_iterations)
        assert hasattr(sampling_iterations, '__len__')
        max_sampling_iterations = max(sampling_iterations)

        x_mu = self.decode(z_mu)['x_mu']

        x_mu_list = []
        z_mu_list = []
        for i in range(0, max_sampling_iterations+1):
            if i in sampling_iterations:
                x_mu_list.append(x_mu)
                z_mu_list.append(z_mu)
            z_mu = self.encode(x_mu)['z_mu']
            x_mu = self.decode(z_mu)['x_mu']

        return dict(x=x_mu_list, z=z_mu_list)

    def get_sampling_iterations_loss(self, dataloader, loss_function, device, *args, **kwargs):
        sampling_iterations = kwargs.get('sampling_iterations', self.sampling_iterations)
        assert hasattr(sampling_iterations, '__len__')
        max_sampling_iterations = max(sampling_iterations)
        loss_sampling_iterations = range(max_sampling_iterations+1)
        self.eval()

        losses_list_dict = []
        with torch.no_grad():
            loss_epoch = {}
            for batch_idx, (x, _) in enumerate(dataloader):
                x = x.to(device)
                z_params = self.encode(x)
                z_r = self.reparameterize(z_params['z_mu'], z_params['z_logvar'])
                samples = self.do_sampling_iterations(z_r['z'],
                                                                   device,
                                                                   sampling_iterations=loss_sampling_iterations)
                x_mu_list, z_mu_list = samples['x'], samples['z']
                for i, iteration in enumerate(loss_sampling_iterations):
                    x_params = self.decode(z_mu_list[i])
                    loss_batch = loss_function(x=x, **x_params, **z_params)
                    for k in loss_batch:
                        loss_epoch[k] = loss_epoch.get(k, 0.) + loss_batch[k]
            losses_list_dict.append(loss_epoch)

        for i in range(len(losses_list_dict)):
            for k in losses_list_dict[i]:
                losses_list_dict[i][k] /= len(dataloader.sampler)
        return dict(losses=losses_list_dict)

    def train_model(self, device, dataloaders, optimizers, losses, logger, **kwargs):
        t0 = time.time()
        best_val_loss = super(VaeModelNormalBase, self).train_model(device, dataloaders, optimizers, losses, logger, **kwargs)
        self.load()
        if logger and logger.flags.get('loss'):
            for data_name, data_loader in dataloaders.items():
                sampling_iterations_dataset_losses = self.get_sampling_iterations_loss(data_loader, losses['vae'], device)
                for i, d in enumerate(sampling_iterations_dataset_losses['losses']):
                    for k, l in d.items():
                        logger.add_scalar('loss/sampling_iterations_%s_%s' % (data_name, k), l, i)
        print('Training of VAE Normal finished in %.2f s' % (time.time()-t0))
        return best_val_loss
