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
    model = mu.get_model(**kwargs['Model'])
    model.load(map_location=device)
    model = model.to(device)
    trainer_loader, tester_loader = mu.get_dataloader(**kwargs['Dataloader'])
    model_parameters_groups = model.get_parameters_groups()
    optimizers = mu.get_optimizers(model_parameters_groups, **kwargs['Optimizers'])
    losses = mu.get_losses(**kwargs['Losses'])
    assert losses.keys() == optimizers.keys()
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
