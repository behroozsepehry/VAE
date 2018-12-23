import torch
from torchvision.utils import save_image

from testers import base


class Tester(base.TesterBase):
    def __init__(self, *args, **kwargs):
            super(Tester, self).__init__()

    def __call__(self, *args, **kwargs):
        assert len(args) == 6
        model, epoch, tester_loader, loss, device, logger = tuple(args)
        verbose = kwargs.get('verbose', False)
        results_path = kwargs.get('path')
        if not results_path:
            if verbose:
                print("\n%s\nNo path is given, terminating test.\n%s" % ('#*10', '#*10'))
            return
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(tester_loader):
                data = data.to(device)
                model_out = model(data)
                test_loss += loss(*((data,)+model_out)).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    x_mu = model_out[0]
                    comparison = torch.cat([data[:n],
                                            x_mu.view(data.size())[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/reconstruction_' + str(epoch) + '.png', nrow=n)

        if verbose:
            test_loss /= len(tester_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))

        if logger:
            logger.add_scalar('data/epoch_test_loss', test_loss, epoch)

        batch_size = tester_loader.batch_size
        with torch.no_grad():
            sample_x, sample_z = model.sample(device, n_samples=batch_size)
            if type(sample_x) != list:
                sample_x_list = [sample_x]
                sample_z_list = [sample_z]
            else:
                sample_x_list = sample_x
                sample_z_list = sample_z
            for i, s in enumerate(sample_x_list):
                save_image(s.cpu().view((batch_size,) + data.size()[1:]),
                           results_path + '/sample_' + str(epoch) + '_' + str(i+1) + '.png')
                if logger:
                    logger.add_embedding(sample_z_list[i].cpu(), global_step=epoch, tag=('data/z_%s'%(i)))
