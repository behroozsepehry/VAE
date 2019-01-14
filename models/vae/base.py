import numpy as np
import torch
from torch import nn

import models.base


class VaeModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__(*args, **kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def _decode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z, **kwargs):
        """separate the _decode and decode allows for modified decode that
            calls _decode, e.g., adding sparsity as part of decoding
        """
        return self._decode(z, **kwargs)

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
        train_batch_losses = {}
        for k in loss_functions:
            optimizers[k].zero_grad()
            loss = loss_functions[k](**model_out)
            loss.backward()
            train_batch_losses[k] = loss.item()
            optimizers[k].step()
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
        return {'model': self.parameters()}
