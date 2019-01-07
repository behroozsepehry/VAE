import numpy as np
import torch
from torch import nn

import models.base


class VaeModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__(*args, **kwargs)

    def encode(self, *args, **kwargs):
        # return tuple
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        # return tuple
        # the first output must be reconstruction
        raise NotImplementedError

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        z_params = self.encode(x)
        z = self.reparameterize(*z_params)
        x_params = self.decode(z)
        return (x,) + x_params + z_params

    def forward_backward(self, *args, **kwargs):
        assert len(args) == 3
        data, loss_functions, optimizers = args
        nol = len(optimizers)
        model_out = self(data)
        train_batch_losses = np.zeros(nol)
        for i in range(nol):
            optimizers[i].zero_grad()
            loss = loss_functions[i](*model_out)
            loss.backward()
            train_batch_losses[i] += loss.item()
            optimizers[i].step()
        return train_batch_losses

    def train_model(self, *args, **kwargs):
        super(VaeModelBase, self).train_model(*args, **kwargs)
        device, trainer_loader, tester_loader, optimizers, losses, logger = args
        self.load()
        if hasattr(self, 'get_sampling_iterations_loss'):
            for data_loader, data_name in [(trainer_loader, 'train'), (tester_loader, 'test')]:
                sampling_iterations_dataset_losses = self.get_sampling_iterations_loss(data_loader, losses[0], device)
                for i, l in enumerate(sampling_iterations_dataset_losses):
                    logger.add_scalar('data/sampling_iterations_%s_losses' % data_name, l, i)
