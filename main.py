import argparse
import yaml
import numpy as np

import torch

from utilities import main_utilities as mu


def construct_objects(**kwargs):
    seed = kwargs.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    device = mu.get_device(**kwargs)
    model, _ = mu.get_models(**kwargs)
    model.load(model.load_path, map_location=device)
    model = model.to(device)
    trainer_loader, tester_loader = mu.get_dataloader(**kwargs)
    if 'gan' in model.name:
        optimizers = [mu.get_optimizer(model.discriminator, **kwargs),
                      mu.get_optimizer(model.generator, **kwargs),]
    else:
        optimizers = [mu.get_optimizer(model, **kwargs),]
    trainer, _ = mu.get_trainers(**kwargs)
    tester, _ = mu.get_testers(**kwargs)
    losses = mu.get_losses(**kwargs)
    assert len(losses) == len(optimizers)
    logger = mu.get_logger(**kwargs)

    if logger:
        logger.add_text('config/config', str(kwargs))

    return device, model, trainer_loader, tester_loader, optimizers, trainer, tester, losses, logger


def train(*args, **kwargs):
    assert len(args) == 9
    device, model, trainer_loader, tester_loader, optimizers, trainer, tester, losses, logger = args
    n_epochs = kwargs['Trainer']['n_epochs']
    if trainer:
        best_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_loss = trainer(model, epoch, [optimizer,], trainer_loader, [loss,], device, logger, **kwargs['Trainer'])
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                model.save(model.save_path, save_data=dict(epoch=epoch))
            if tester:
                test_loss = tester(model, epoch, tester_loader, loss, device, logger, **kwargs['Tester'])

    if not (trainer and n_epochs):
        test_loss = tester(model, 0, tester_loader, loss, device, logger, **kwargs['Tester'])

    model.load(model.save_path)
    for data_loader, data_name in [(trainer_loader, 'train'), (tester_loader, 'test')]:
        sampling_iterations_dataset_losses = model.get_sampling_iterations_loss(data_loader, loss, device)
        for i, l in enumerate(sampling_iterations_dataset_losses):
            logger.add_scalar('data/sampling_iterations_%s_losses' % data_name, l, i)


def  main():
    parser = argparse.ArgumentParser(description='Variational Auto Encoder Experiments')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/conf_mnist_1.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        kwargs = yaml.load(f)
        objects = construct_objects(**kwargs)
        device, model, trainer_loader, tester_loader, optimizer, trainer, tester, loss, logger = objects
        train(*objects, **kwargs)
        if logger:
            logger.close()


if __name__ == '__main__':
    main()
