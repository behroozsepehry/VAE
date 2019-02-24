import torch

from evaluator import kl_base
from losses.gan import kl_generator


class Evaluator(kl_base.Evaluator):
    def __init__(self, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)

    def __call__(self):
        self.train()
        n_samples = 0
        result = {}
        eval_loss_func = kl_generator.Loss()
        with torch.no_grad():
            for dataloader_name, dataloader in self.dataloaders.items():
                eval = 0.
                for batch_idx, (x, _) in enumerate(dataloader):
                    x = x.to(self.device)
                    y_real = self.discriminator(x)
                    eval += eval_loss_func._compute_losses(y_real)
                    n_samples += x.size(0)
                    if n_samples >= self.n_samples:
                        break
                eval /= n_samples
                result[dataloader_name] = eval
        return result
