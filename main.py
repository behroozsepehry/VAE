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
    models = mu.get_models(**kwargs)
    model = models[0]
    model.load(map_location=device)
    model = model.to(device)
    trainer_loader, tester_loader = mu.get_dataloader(**kwargs)
    if 'gan' in str(type(model)):
        optimizers = [mu.get_optimizer(model.discriminator, **kwargs),
                      mu.get_optimizer(model.generator, **kwargs),]
    else:
        optimizers = [mu.get_optimizer(model, **kwargs),]
    trainer = mu.get_trainers(**kwargs)[0]
    tester = mu.get_testers(**kwargs)[0]
    losses = mu.get_losses(**kwargs)
    assert len(losses) == len(optimizers)
    logger = mu.get_logger(**kwargs)

    if logger:
        logger.add_text('config/config', str(kwargs))

    return device, model, trainer_loader, tester_loader, optimizers, trainer, tester, losses, logger


def train(*args, **kwargs):
    assert len(args) == 9
    device, model, trainer_loader, tester_loader, optimizers, trainer, tester, losses, logger = args
    n_epochs = kwargs['Trainer'][0]['n_epochs']
    if trainer:
        best_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_loss = trainer(model, epoch, optimizers, trainer_loader, losses, device, logger, **kwargs['Trainer'][0])
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                model.save(model.save_path, save_data=dict(epoch=epoch))
            if tester:
                test_loss = tester(model, epoch, tester_loader, losses, device, logger, **kwargs['Tester'][0])

    if not (trainer and n_epochs):
        test_loss = tester(model, 0, tester_loader, losses, device, logger, **kwargs['Tester'][0])

    model.load(model.save_path)
    if hasattr(model, 'get_sampling_iterations_loss'):
        for data_loader, data_name in [(trainer_loader, 'train'), (tester_loader, 'test')]:
            sampling_iterations_dataset_losses = model.get_sampling_iterations_loss(data_loader, losses[0], device)
            for i, l in enumerate(sampling_iterations_dataset_losses):
                logger.add_scalar('data/sampling_iterations_%s_losses' % data_name, l, i)


def main():
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
