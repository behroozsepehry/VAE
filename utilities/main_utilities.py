import torch
from torch import optim
from torchvision import datasets, transforms

from utilities import general_utilities


def get_instances(folder, instance_type, **kwargs):
    instance_list = []
    for i, kwargs_instance in enumerate(kwargs[instance_type]):
        instance_filename = kwargs_instance['name']
        if not instance_filename:
            instance = None
        else:
            module = general_utilities.import_from_path(folder+instance_filename+'.py')
            instance = getattr(module, instance_type)(**kwargs_instance)
        instance_list.append(instance)
    return instance_list


def get_models(**kwargs):
    return get_instances('models/', 'Model', **kwargs)


def get_optimizer(model_, **kwargs):
    optimizer_name = kwargs['Optimizer']['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(model_.parameters(), **kwargs['Optimizer']['args'])
    return optimizer


def get_dataloader(**kwargs):
    dataset_path = kwargs['Dataloader']['path']
    dataset_name = kwargs['Dataloader']['name']
    dataset_constructor = getattr(datasets, dataset_name)
    trainer_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=True, download=True,  transform=transforms.ToTensor()),
        **kwargs['Dataloader']['args'])
    tester_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=False, transform=transforms.ToTensor()),
        **kwargs['Dataloader']['args'])

    return trainer_loader, tester_loader


def get_trainers(**kwargs):
    return get_instances('trainers/', 'Trainer', **kwargs)


def get_testers(**kwargs):
    return get_instances('testers/', 'Tester', **kwargs)


def get_losses(**kwargs):
    return get_instances('losses/', 'Loss', **kwargs)


def get_device(**kwargs):
    device_name = kwargs['Device']['name']
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device_name_2 = 'cpu'
        device = torch.device(device_name_2)
        if device_name_2 != device_name:
            print('Warning: device \'%s\' not available, using device \'%s\' instead'% (device_name, device_name_2))
    return device


def get_logger(**kwargs):
    if kwargs['Logger']['name'] == 'tensorboard':
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(**kwargs['Logger']['args'])
    elif not kwargs['Logger']['name']:
        logger = None
    else:
        raise NotImplementedError
    return logger
