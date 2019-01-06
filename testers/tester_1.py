import numpy as np
import torch
from torchvision.utils import save_image

from testers import base


class Tester(base.TesterBase):
    def __init__(self, *args, **kwargs):
            super(Tester, self).__init__()

    def __call__(self, *args, **kwargs):
        assert len(args) == 6
        model, epoch, tester_loader, losses, device, logger = tuple(args)
        verbose = kwargs.get('verbose', False)
        results_path = kwargs.get('path')
        if not results_path:
            if verbose:
                print("\n%s\nNo path is given, terminating test.\n%s" % ('#*10', '#*10'))
            return
        model.eval()

        nol = len(losses)

        test_losses = np.zeros(nol)
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(tester_loader):
                data = data.to(device)
                model_out = model(data)
                for il in range(nol):
                    test_losses[il] += losses[il](*((data,)+model_out)).item()
                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    x_mu = model_out[0]
                    comparison = torch.cat([data[:n],
                                            x_mu.view(data.size())[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/reconstruction_' + str(epoch) + '.png', nrow=n)

        if verbose:
            test_losses /= len(tester_loader.dataset)
            print('====> Test set loss: {}'.format(test_losses))

        if logger:
            for i, loss_ in enumerate(test_losses):
                logger.add_scalar('data/epoch_test_loss_%s' % i, loss_, epoch)

        batch_size = tester_loader.batch_size
        if hasattr(model, 'generate'):
            with torch.no_grad():
                sample_x, sample_z = model.generate(device, n_samples=batch_size)
                if type(sample_x) != list:
                    sample_x_list = [sample_x]
                    sample_z_list = [sample_z]
                else:
                    sample_x_list = sample_x
                    sample_z_list = sample_z
                for i, s in enumerate(sample_x_list):
                    x_images = s.cpu().view((batch_size,) + data.size()[1:])
                    save_image(x_images,
                               results_path + '/sample_' + str(epoch) + '_' + str(i+1) + '.png')
                    if logger:
                        logger.add_embedding(sample_z_list[i].cpu(),
                                             tag=('data/z_%s_%s'%(epoch, i)),
                                             label_img=x_images)

        return test_losses
