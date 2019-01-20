import numpy as np
import time

import torch
from torch import nn
from torchvision.utils import save_image
from utilities import general_utilities as gu


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ModelBase, self).__init__()
        self.save_path = kwargs.get('save_path')
        self.load_path = kwargs.get('load_path')
        self.name = kwargs.get('name')
        self.train_args = kwargs.get('train_args', {})
        self.evaluate_args = kwargs.get('evaluate_args', {})

    def load(self, path=None, **kwargs):
        if not path:
            path = self.load_path
        if path:
            data = torch.load(path, map_location=kwargs.get('map_location'))
            self.load_state_dict(data['state_dict'])

    def save(self, path=None, **kwargs):
        save_data = kwargs.get('save_data', {})
        assert type(save_data) == dict
        if not path:
            path = self.save_path
        if path:
            torch.save(dict(**save_data, state_dict=self.state_dict()), path)

    def forward_backward(self, *args, **kwargs):
        raise NotImplementedError

    def train_epoch(self, epoch, optimizers, trainer_loader, loss_functions, device, logger, **kwargs):
        t0 = time.time()

        assert hasattr(optimizers, '__len__')  # check if we have list, tuple, etc of optimizers
        assert hasattr(loss_functions, '__len__')  # check if we have list, tuple, etc of loss_functions
        assert len(optimizers) == len(loss_functions)

        log_interval = kwargs.get('log_interval', self.train_args.get('log_interval', 1))
        verbose = kwargs.get('verbose', self.train_args.get('verbose', False))

        self.train()
        train_losses = 0.
        for batch_idx, (x, _) in enumerate(trainer_loader):
            x = x.to(device)
            model_out = self.forward_backward(x, loss_functions, optimizers)
            loss_values = np.array([x.item() for x in model_out['losses'].values()])
            loss_keys = model_out['losses'].keys()

            train_losses += loss_values
            if verbose:
                if batch_idx % log_interval == 0:
                    loss_avg_key_value = list(zip(loss_keys, loss_values / len(x)))
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch, batch_idx * len(x), len(trainer_loader.sampler),
                        100. * batch_idx / len(trainer_loader),
                        loss_avg_key_value))

        epoch_train_losses = train_losses / len(trainer_loader.sampler)
        epoch_train_loss_avg_key_value = list(zip(loss_keys, epoch_train_losses))
        if verbose:
            print('====> Epoch: {} Average loss: {}'.format(
                epoch, epoch_train_loss_avg_key_value))
            print('Time: %.2f s' % (time.time()-t0))

        if logger:
            for (k, v) in epoch_train_loss_avg_key_value:
                logger.add_scalar('data/epoch_train_loss_%s' % k, v, epoch)

        return dict(losses=epoch_train_losses)

    def train_model(self, device, dataloaders, optimizers, losses, logger, **kwargs):
        t0 = time.time()
        n_epochs = kwargs.get('n_epochs', self.train_args['n_epochs'])

        test_loss = self.evaluate_epoch(0, dataloaders['test'], losses, device, logger, name='test')['losses'][0]
        val_loss = self.evaluate_epoch(0, dataloaders['val'], losses, device, logger, name='val')['losses'][0]

        best_val_loss = val_loss
        validated_test_loss = test_loss
        validated_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch, optimizers, dataloaders['train'], losses, device, logger)['losses'][0]
            test_loss = self.evaluate_epoch(epoch, dataloaders['test'], losses, device, logger, name='test')['losses'][0]
            val_loss = self.evaluate_epoch(epoch, dataloaders['val'], losses, device, logger, name='val')['losses'][0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                validated_test_loss = test_loss
                validated_train_loss = train_loss
                self.save(save_data=dict(epoch=epoch,
                                         train_loss=train_loss, val_loss=val_loss, test_loss=test_loss))
        print('Validated train/val/test loss: %.4f/%.4f/%.4f' % (validated_train_loss, best_val_loss, validated_test_loss))
        print('Training finished in %.2f s' % (time.time()-t0))
        return best_val_loss

    def evaluate_epoch(self, epoch, tester_loader, losses, device, logger, **kwargs):
        t0 = time.time()

        verbose = kwargs.get('verbose', self.evaluate_args.get('verbose', False))
        results_path = kwargs.get('path', self.evaluate_args.get('path'))
        name = kwargs.get('name', 'test')
        if not results_path:
            if verbose:
                print("\n%s\nNo path is given, terminating test.\n%s" % ('#*10', '#*10'))
            return

        self.eval()

        test_loss_vals = 0.
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tester_loader):
                x = x.to(device)
                model_out = self(x)
                batch_losses = {}
                for k in losses:
                    losses_v = losses[k](**model_out)
                    batch_losses.update(gu.append_key_dict(losses_v, k+'_'))

                test_loss_vals += np.array([x.item() for x in batch_losses.values()])

                if 'vae' in self.name and batch_idx == 0:
                    n = min(x.size(0), 8)
                    x_mu = model_out['x_mu']
                    comparison = torch.cat([x[:n],
                                            x_mu.view(x.size())[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/reconstruction_' + name + '_' + str(epoch) + '.png', nrow=n)

        test_losses_avg = test_loss_vals / len(tester_loader.sampler)
        test_losses_avg_key_value = list(zip(batch_losses.keys(), test_losses_avg))
        if verbose:
            print('====> {} set loss: {}'.format(name, test_losses_avg_key_value))

        if logger:
            for k, v in test_losses_avg_key_value:
                logger.add_scalar('data/epoch_%s_loss_%s' % (name, k), v, epoch)

        batch_size = tester_loader.batch_size
        if hasattr(self, 'generate'):
            with torch.no_grad():
                samples = self.generate(device, n_samples=batch_size)
                sample_x, sample_z = samples['x'], samples['z']
                if type(sample_x) != list:
                    sample_x_list = [sample_x]
                    sample_z_list = [sample_z]
                else:
                    sample_x_list = sample_x
                    sample_z_list = sample_z
                for i, s in enumerate(sample_x_list):
                    x_images = s.cpu().view((batch_size,) + x.size()[1:])
                    save_image(x_images,
                               results_path + '/sample_' + str(epoch) + '_' + str(i+1) + '.png')
                    if logger:
                        logger.add_embedding(sample_z_list[i].cpu(),
                                             tag=('data/z_%s_%s_%s' % (name, epoch, i)),
                                             label_img=x_images)

        if verbose:
            print('Time: %.2f s' % (time.time()-t0))

        return dict(losses=test_losses_avg)

    def evaluate_model(self, *args, **kwargs):
        return self.evaluate_epoch(*args, **kwargs)

    def get_parameters_groups(self):
        raise NotImplementedError

