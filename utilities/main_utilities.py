import torch
from torch import optim
from torchvision import datasets, transforms

from utilities import general_utilities


def get_instance(folder, instance_type, **kwargs):
    instance_filename = kwargs[instance_type]['name']
    module = general_utilities.import_from_path(folder+instance_filename+'.py')
    instance = getattr(module, instance_type)(**kwargs[instance_type])
    return instance


def get_model(**kwargs):
    return get_instance('models/', 'Model', **kwargs)


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


def get_trainer(**kwargs):
    return get_instance('trainers/', 'Trainer', **kwargs)


def get_tester(**kwargs):
    return get_instance('testers/', 'Tester', **kwargs)


def get_loss(**kwargs):
    return get_instance('losses/', 'Loss', **kwargs)
