import numpy as np
import torch
from torch import nn

import models.base


class VaeModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__(*args, **kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        z_params = self.encode(x)
        z_r = self.reparameterize(**z_params)
        x_params = self.decode(z_r['z'])
        return dict(x=x, **x_params, **z_params)

    def forward_backward(self, x, loss_functions, optimizers, **kwargs):
        nol = len(optimizers)
        model_out = self(x)
        train_batch_losses = np.zeros(nol)
        for i in range(nol):
            optimizers[i].zero_grad()
            loss = loss_functions[i](**model_out)
            loss.backward()
            train_batch_losses[i] += loss.item()
            optimizers[i].step()
        return dict(losses=train_batch_losses)

    def train_model(self, device, trainer_loader, tester_loader, optimizers, losses, logger, **kwargs):
        super(VaeModelBase, self).train_model(device, trainer_loader, tester_loader, optimizers, losses, logger, **kwargs)
        self.load()
        if hasattr(self, 'get_sampling_iterations_loss'):
            for data_loader, data_name in [(trainer_loader, 'train'), (tester_loader, 'test')]:
                sampling_iterations_dataset_losses = self.get_sampling_iterations_loss(data_loader, losses[0], device)
                for i, l in enumerate(sampling_iterations_dataset_losses['losses']):
                    logger.add_scalar('data/sampling_iterations_%s_losses' % data_name, l, i)

    def get_parameters_groups(self):
        return [self.parameters()]
