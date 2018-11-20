import torch
from torchvision.utils import save_image

from testers import base


class Tester(base.TesterBase):
    def __init__(self, *args, **kwargs):
            super(Tester, self).__init__()

    def __call__(self, *args, **kwargs):
        assert len(args) == 5
        model, epoch, tester_loader, loss, device = tuple(args)
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
                    recon_batch, mu = model_out[0:2]
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(data.size())[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/reconstruction_' + str(epoch) + '.png', nrow=n)

        if verbose:
            test_loss /= len(tester_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))

        batch_size = tester_loader.batch_size
        with torch.no_grad():
            sample = torch.randn((batch_size, mu.size(1))).to(device)
            sample = model.decode(sample)[0].cpu()
            save_image(sample.view((batch_size,) + data.size()[1:]),
                       results_path + '/sample_' + str(epoch) + '.png')