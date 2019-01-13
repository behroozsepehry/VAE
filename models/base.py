import numpy as np

import torch
from torch import nn
from torchvision.utils import save_image


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
            train_losses += model_out['losses']
            if verbose:
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch, batch_idx * len(x), len(trainer_loader.dataset),
                        100. * batch_idx / len(trainer_loader),
                        model_out['losses'] / len(x)))

        epoch_train_losses = train_losses / len(trainer_loader.dataset)
        if verbose:
            print('====> Epoch: {} Average loss: {}'.format(
                epoch, epoch_train_losses))
        if logger:
            for i, loss_ in enumerate(epoch_train_losses):
                logger.add_scalar('data/epoch_train_loss_%s' % i, loss_, epoch)

        return dict(losses=epoch_train_losses)

    def train_model(self, device, trainer_loader, tester_loader, optimizers, losses, logger, **kwargs):
        n_epochs = kwargs.get('n_epochs', self.train_args['n_epochs'])

        test_loss = self.evaluate_epoch(0, tester_loader, losses, device, logger)
        best_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_losses = self.train_epoch(epoch, optimizers, trainer_loader, losses, device, logger)
            train_loss = train_losses['losses'][0]
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.save(save_data=dict(epoch=epoch))
            test_loss = self.evaluate_epoch(epoch, tester_loader, losses, device, logger)

    def evaluate_epoch(self, epoch, tester_loader, losses, device, logger, **kwargs):
        verbose = kwargs.get('verbose', self.evaluate_args.get('verbose', False))
        results_path = kwargs.get('path', self.evaluate_args.get('path'))
        if not results_path:
            if verbose:
                print("\n%s\nNo path is given, terminating test.\n%s" % ('#*10', '#*10'))
            return
        self.eval()

        nol = len(losses)

        test_losses = np.zeros(nol)
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tester_loader):
                x = x.to(device)
                model_out = self(x)
                for il in range(nol):
                    test_losses[il] += losses[il](**model_out).item()
                if 'vae' in self.name and batch_idx == 0:
                    n = min(x.size(0), 8)
                    x_mu= model_out['x_mu']
                    comparison = torch.cat([x[:n],
                                            x_mu.view(x.size())[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/reconstruction_' + str(epoch) + '.png', nrow=n)

        if verbose:
            test_losses /= len(tester_loader.dataset)
            print('====> Test set loss: {}'.format(test_losses))

        if logger:
            for i, loss_ in enumerate(test_losses):
                logger.add_scalar('data/epoch_test_loss_%s' % i, loss_, epoch)

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
                                             tag=('data/z_%s_%s'%(epoch, i)),
                                             label_img=x_images)

        return dict(losses=test_losses)

    def evaluate_model(self, *args, **kwargs):
        return self.evaluate_epoch(*args, **kwargs)

    def get_parameters_groups(self):
        raise NotImplementedError

