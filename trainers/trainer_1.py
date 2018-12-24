from trainers import base


class Trainer(base.TrainerBase):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__()

    def __call__(self, *args, **kwargs):
        assert len(args) == 7
        model, epoch, optimizer, trainer_loader, loss_function, device, logger = tuple(args)
        log_interval = kwargs.get('log_interval', 1)
        verbose = kwargs.get('verbose', False)
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(trainer_loader):
            data = data.to(device)
            optimizer.zero_grad()
            model_out = model(data)
            loss = loss_function(*((data,)+model_out))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if verbose:
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(trainer_loader.dataset),
                        100. * batch_idx / len(trainer_loader),
                        loss.item() / len(data)))

        epoch_train_loss = train_loss / len(trainer_loader.dataset)
        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, epoch_train_loss))
        if logger:
            logger.add_scalar('data/epoch_train_loss', epoch_train_loss, epoch)

        return epoch_train_loss

