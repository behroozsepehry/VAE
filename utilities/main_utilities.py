import importlib.util

import torch
from torch import optim
from torchvision import datasets, transforms

from models import mnist_simple_fc_1
from trainers import trainer_1
from testers import tester_1
from losses import vae_1

def get_model(**kwargs):
    models_dict = {
        'mnist_simple_fc_1': mnist_simple_fc_1.MnistSimpleFc1,
    }
    model_name = kwargs['model']['name']
    model = models_dict[model_name](**kwargs['model'])
    return model


def get_optimizer(model_, **kwargs):
    optimizer_name = kwargs['optimizer']['name']
    optimizer_constructor = getattr(optim, optimizer_name)
    optimizer = optimizer_constructor(model_.parameters(), **kwargs['optimizer']['args'])
    return optimizer


def get_dataloader(**kwargs):
    dataset_path = kwargs['dataloader']['path']
    dataset_name = kwargs['dataloader']['name']

    dataset_constructor = getattr(datasets, dataset_name)
    trainer_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=True, download=True,  transform=transforms.ToTensor()),
        **kwargs['dataloader']['args'])
    tester_loader = torch.utils.data.DataLoader(
        dataset_constructor(dataset_path, train=False, transform=transforms.ToTensor()),
        **kwargs['dataloader']['args'])

    return trainer_loader, tester_loader


def get_trainer(**kwargs):
    trainers_dict = {
        'trainer_1': trainer_1,
    }
    trainer_name = kwargs['trainer']['name']
    trainer = trainers_dict[trainer_name].train
    return trainer


def get_tester(**kwargs):
    testers_dict = {
        'tester_1': tester_1,
    }
    tester_name = kwargs['tester']['name']
    tester = testers_dict[tester_name].test
    return tester


def get_loss_function(**kwargs):
    loss_name = kwargs['loss']['name']
    spec = importlib.util.spec_from_file_location("loss_spec", "losses/"+loss_name+'.py')
    loss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loss_module)

    loss = loss_module.Loss(**kwargs['loss'].get('args', {}))
    return loss

