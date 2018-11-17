import argparse
import yaml

import torch

from utilities import main_utilities as mu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder Experiments')
    parser.add_argument('--conf-path', '-c', type=str, default='confs/conf_mnist_1.yaml', metavar='N',
                        help='configuration file path')
    args = parser.parse_args()
    with open(args.conf_path, 'rb') as f:
        kwargs = yaml.load(f)

        device_name = kwargs['device']['name']
        device = torch.device(device_name)

        model = mu.get_model(**kwargs).to(device)
        trainer_loader, tester_loader = mu.get_dataloader(**kwargs)
        optimizer = mu.get_optimizer(model, **kwargs)
        trainer = mu.get_trainer(**kwargs)
        tester = mu.get_tester(**kwargs)
        loss = mu.get_loss_function(**kwargs)

        n_epochs = kwargs['trainer']['n_epochs']
        for epoch in range(1, n_epochs + 1):
            trainer(model, epoch, optimizer, trainer_loader, loss, device, **kwargs['trainer'])
            tester(model, epoch, tester_loader, loss, device, **kwargs['tester'])

