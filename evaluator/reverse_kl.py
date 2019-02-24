import torch

from evaluator import kl_base
from losses.gan import reverse_kl_generator


class Evaluator(kl_base.Evaluator):
    def __init__(self, *args, **kwargs):
        super(Evaluator, self).__init__(*args, **kwargs)

    def __call__(self):
        self.train()
        with torch.no_grad():
            samples = self.generator.generate(self.device, n_samples=self.n_samples)
            y_fake = self.discriminator(samples['x'])
            eval_loss_func = reverse_kl_generator.Loss()
            result = eval_loss_func._compute_losses(y_fake) / self.n_samples
        return result
