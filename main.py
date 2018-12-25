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
    model = mu.get_model(**kwargs)
    model.load(model.load_path, map_location=device)
    model = model.to(device)
    trainer_loader, tester_loader = mu.get_dataloader(**kwargs)
    optimizer = mu.get_optimizer(model, **kwargs)
    trainer = mu.get_trainer(**kwargs)
    tester = mu.get_tester(**kwargs)
    loss = mu.get_loss(**kwargs)
    logger = mu.get_logger(**kwargs)

    if logger:
        logger.add_text('config/config', str(kwargs))

    return device, model, trainer_loader, tester_loader, optimizer, trainer, tester, loss, logger


def train(*args, **kwargs):
    assert len(args) == 9
    device, model, trainer_loader, tester_loader, optimizer, trainer, tester, loss, logger = args
    n_epochs = kwargs['Trainer']['n_epochs']
    if trainer:
        best_train_loss = np.inf
        for epoch in range(1, n_epochs + 1):
            train_loss = trainer(model, epoch, optimizer, trainer_loader, loss, device, logger, **kwargs['Trainer'])
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                model.save(model.save_path, save_data=dict(epoch=epoch))
            if tester:
                test_loss = tester(model, epoch, tester_loader, loss, device, logger, **kwargs['Tester'])

    if not (trainer and n_epochs):
        test_loss = tester(model, epoch, tester_loader, loss, device, logger, **kwargs['Tester'])

    model.load(model.save_path)
    sampling_iterations_train_losses = model.get_sampling_iterations_loss(trainer_loader, loss, device)
    for i, l in enumerate(sampling_iterations_train_losses):
        logger.add_scalar('data/sampling_iterations_train_losses', l, i)


if __name__ == '__main__':
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
