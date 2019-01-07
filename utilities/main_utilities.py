import torch
from torch import optim
from torchvision import datasets, transforms

from utilities import general_utilities


def get_instances(folder, instance_type, *args):
    instance_list = []
    for i, kwargs_instance in enumerate(args):
        instance_filename = kwargs_instance['name']
        if not instance_filename:
            instance = None
        else:
            module = general_utilities.import_from_path(folder+instance_filename+'.py')
            instance = getattr(module, instance_type)(**kwargs_instance)
        instance_list.append(instance)
    return instance_list


def get_models(*args):
    return get_instances('models/', 'Model', *args)


def get_optimizer(parameters, **kwargs):
    optimizer_name = kwargs['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(parameters, **kwargs['args'])
    return optimizer


def get_optimizers(parameters_groups, *args):
    assert len(parameters_groups) == len(args)
    return [get_optimizer(parameters_groups[i], **args[i]) for i in range(len(args))]


def get_dataloader(**kwargs):
    dataset_path = kwargs['path']
    dataset_name = kwargs['name']
    dataset_constructor = getattr(datasets, dataset_name)
    trainer_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=True, download=True,  transform=transforms.ToTensor()),
        **kwargs['args'])
    tester_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=False, transform=transforms.ToTensor()),
        **kwargs['args'])

    return trainer_loader, tester_loader


def get_trainers(*args):
    return get_instances('trainers/', 'Trainer', *args)


def get_testers(*args):
    return get_instances('testers/', 'Tester', *args)


def get_losses(*args):
    return get_instances('losses/', 'Loss', *args)


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
    elif not kwargs['name']:
        logger = None
    else:
        raise NotImplementedError
    return logger
