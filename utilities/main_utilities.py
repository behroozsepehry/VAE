import numpy as np

import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utilities import general_utilities


def get_instance(folder, instance_type, **kwargs):
    kwargs_instance = kwargs
    instance_filename = kwargs_instance['name']
    if not instance_filename:
        instance = None
    else:
        module = general_utilities.import_from_path(folder+instance_filename+'.py')
        instance = getattr(module, instance_type)(**kwargs_instance)
    return instance


def get_instances_dict(folder, instance_type, **kwargs):
    return {k: get_instance(folder, instance_type, **v) for k, v in kwargs.items()}


def get_model(**kwargs):
    return get_instance('models/', 'Model', **kwargs)


def get_optimizer(parameters, **kwargs):
    optimizer_name = kwargs['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(parameters, **kwargs['args'])
    return optimizer


def get_optimizers(parameters_groups, **kwargs):
    assert parameters_groups.keys() == kwargs.keys()
    return {k: get_optimizer(parameters_groups[k], **v) for k, v in kwargs.items()}


def get_dataloaders(**kwargs):
    dataset_path = kwargs['path']
    dataset_name = kwargs['name']
    dataset_constructor = getattr(datasets, dataset_name)

    dataset_eval = dataset_constructor(dataset_path, train=False, download=True, transform=transforms.ToTensor())
    dataset_eval_size = len(dataset_eval)
    idxs = range(dataset_eval_size)
    val_ratio = kwargs['ratio'].get('val', 1. - kwargs['ratio'].get('test', 1.))
    split_idx = int(np.floor(val_ratio * dataset_eval_size))
    val_idxs = idxs[:split_idx]
    test_idxs = idxs[split_idx:]
    val_sampler = SubsetRandomSampler(val_idxs)
    test_sampler = SubsetRandomSampler(test_idxs)

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            dataset_constructor(dataset_path, train=True, download=True,  transform=transforms.ToTensor()),
            **kwargs['args']),
        'test': torch.utils.data.DataLoader(
            dataset_eval,
            sampler=test_sampler,
            **dict(kwargs['args'], shuffle=False)),
        'val': torch.utils.data.DataLoader(
            dataset_eval,
            sampler=val_sampler,
            **dict(kwargs['args'], shuffle=False)),
    }

    return dataloaders


def get_losses(**kwargs):
    return get_instances_dict('losses/', 'Loss', **kwargs)


def get_device(**kwargs):
    device_name = kwargs['name']
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device_name_2 = 'cpu'
        device = torch.device(device_name_2)
        if device_name_2 != device_name:
            print('Warning: device \'%s\' not available, using device \'%s\' instead'% (device_name, device_name_2))
    return device


def get_logger(**kwargs):
    if kwargs['name'] == 'tensorboard':
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(**kwargs['args'])
        logger.flags = kwargs.get('flags', {})
    elif not kwargs['name']:
        logger = None
    else:
        raise NotImplementedError
    return logger
