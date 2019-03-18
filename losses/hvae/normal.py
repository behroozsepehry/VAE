from typing import Dict, List

import numpy as np
import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class Loss(base.LossBase):
    def __init__(self):
        super(Loss, self).__init__()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_hat_params: Dict[str, torch.Tensor],
                 p_params: List[Dict[str, torch.Tensor]], q_params: List[Dict[str, torch.Tensor]],
                 **kwargs):
        x_mu = x_hat_params['mu']
        x_logvar = x_hat_params['logvar']

        mse = 0.5 * (torch.sum(x_logvar.exp() + (((x_mu - x.view(x_mu.size())).pow(2))/(x_logvar.exp()))))
        kld = 0.
        for i in range(len(p_params)):
            q_mu, q_logvar = q_params[i]['mu'], q_params[i]['logvar']
            p_mu, p_logvar = p_params[i]['mu'], p_params[i]['logvar']
            kld += -0.5 * torch.sum((q_logvar - p_logvar).exp() + (p_mu - q_mu).pow(2)/(p_logvar.exp()) + p_logvar - q_logvar)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

        return dict(loss=bce+kld, bce=bce, kld=kld)
