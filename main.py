import argparse
import yaml
import numpy as np

import torch

from utilities import main_utilities as mu


def construct_objects(**kwargs):
    seed = kwargs.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    device = mu.get_device(**kwargs['Device'])
    models = mu.get_models(*kwargs['Models'])
    model, = models
    model.load(map_location=device)
    model = model.to(device)
    trainer_loader, tester_loader = mu.get_dataloader(**kwargs['Dataloader'])
    if 'gan' in model.name:
        optimizers = [mu.get_optimizer(model.discriminator, **kwargs['Optimizers'][0]),
                      mu.get_optimizer(model.generator, **kwargs['Optimizers'][1]),]
    else:
        optimizers = [mu.get_optimizer(model, **kwargs['Optimizers'][0]),]
    losses = mu.get_losses(*kwargs['Losses'])
    assert len(losses) == len(optimizers)
    logger = mu.get_logger(**kwargs['Logger'])

    if logger:
        logger.add_text('config/config', str(kwargs))

    return device, model, trainer_loader, tester_loader, optimizers, losses, logger

def main():
    parser = argparse.ArgumentParser(description='Variational Auto Encoder Experiments')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/conf_mnist_vae_1.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        kwargs = yaml.load(f)
        objects = construct_objects(**kwargs)
        device, model, trainer_loader, tester_loader, optimizers, losses, logger = objects
        model.train_model(device, trainer_loader, tester_loader, optimizers, losses, logger)
        if logger:
            logger.close()


if __name__ == '__main__':
    main()
