import argparse
import yaml
import itertools
import copy
import numpy as np

import torch

from utilities import main_utilities as mu
from utilities import general_utilities as gu


def construct_objects(settings):
    seed = settings.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    device = mu.get_device(**settings['Device'])
    model = mu.get_model(**settings['Model'])
    model.load(map_location=device)
    model = model.to(device)
    dataloaders = mu.get_dataloaders(**settings['Dataloaders'])
    model_parameters_groups = model.get_parameters_groups()
    optimizers = mu.get_optimizers(model_parameters_groups, **settings['Optimizers'])
    losses = mu.get_losses(**settings['Losses'])
    assert losses.keys() == optimizers.keys()
    logger = mu.get_logger(**settings['Logger'])

    if logger and logger.flags.get('conf'):
        logger.add_text('conf/conf', str(settings))

    return device, model, dataloaders, optimizers, losses, logger


def train(settings):
    objects = construct_objects(settings)
    device, model, dataloaders, optimizers, losses, logger = objects
    result = model.train_model(device, dataloaders, optimizers, losses, logger)
    if logger:
        logger.close()
    return result


def hypertune(settings):
    hypertune_settings = settings.get('Hypertune', {})
    hypertune_params_settings = hypertune_settings.get('params', {})
    hypertune_params_settings_list_sizes = gu.get_list_size_of_dict_of_lists(hypertune_params_settings)
    hypertune_params_settings_list_sizes_lists = [range(s) for s in hypertune_params_settings_list_sizes]
    all_possible_choices = itertools.product(*hypertune_params_settings_list_sizes_lists)

    best_choice = None
    best_val = np.inf
    for choice in all_possible_choices:
        train_settings = copy.deepcopy(settings)
        gu.choose_1_from_dict_of_lists(hypertune_params_settings, choice, train_settings)
        val = train(train_settings)
        if val < best_val:
            best_val = val
            best_choice = copy.deepcopy(choice)

    best_hyperparams = copy.deepcopy(hypertune_params_settings)
    gu.choose_1_from_dict_of_lists(hypertune_params_settings, best_choice, best_hyperparams)

    return dict(best_hyperparams=best_hyperparams, best_val=best_val)


def main():
    parser = argparse.ArgumentParser(description='Variational Auto Encoder Experiments')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/conf_mnist_vae_sparse_1.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        settings = yaml.load(f)

    if settings['function'] == 'train':
        result = train(settings)
    elif settings['function'] == 'hypertune':
        result = hypertune(settings)

    print('result: %s' % (result))


if __name__ == '__main__':
    main()
