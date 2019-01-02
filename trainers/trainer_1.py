import numpy as np

from trainers import base


class Trainer(base.TrainerBase):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__()

    def __call__(self, *args, **kwargs):
        assert len(args) == 7
        model, epoch, optimizers, trainer_loader, loss_functions, device, logger = tuple(args)
        assert hasattr(optimizers, '__len__')  # check if we have list, tuple, etc of optimizers
        assert hasattr(loss_functions, '__len__')  # check if we have list, tuple, etc of loss_functions
        assert len(optimizers) == len(loss_functions)
        nol = len(optimizers)

        log_interval = kwargs.get('log_interval', 1)
        verbose = kwargs.get('verbose', False)
        model.train()
        train_losses = np.zeros(nol)
        for batch_idx, (data, _) in enumerate(trainer_loader):
            data = data.to(device)
            model_out = model(data)
            train_batch_losses = np.zeros(nol)
            for i in range(nol):
                optimizers[i].zero_grad()
                loss = loss_functions[i](*((data,) + model_out))
                loss.backward()
                train_batch_losses[i] += loss.item()
                optimizers[i].step()
            train_losses += train_batch_losses
            if verbose:
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch, batch_idx * len(data), len(trainer_loader.dataset),
                        100. * batch_idx / len(trainer_loader),
                        train_batch_losses / len(data)))

        epoch_train_losses = train_losses / len(trainer_loader.dataset)
        if verbose:
            print('====> Epoch: {} Average loss: {}'.format(
                epoch, epoch_train_losses))
        if logger:
            for i, l in enumerate(epoch_train_losses):
                logger.add_scalar('data/epoch_train_loss_%s' % i, l, epoch)

        return epoch_train_losses

