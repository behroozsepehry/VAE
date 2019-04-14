from typing import Dict, List
import torch
import torch.nn
from torch.nn import functional as F

from losses import base


class Loss(base.LossBase):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()

    # Reconstruction + KL divergence losses summed over all elements and batch
    def __call__(self, x, x_hat_params: Dict[str, torch.Tensor],
                 p_params: List[Dict[str, torch.Tensor]], q_params: List[Dict[str, torch.Tensor]],
                 **kwargs):
        """
        Assumes that z = [z1, z2, ..., zn] with graphical model z1->z2....->zn
        :param x: input
        :param x_hat_params: parameters of the distribution of output
        :param p_params: the parameters of distributions of z=[z1, z2, ..., zn] for p (prior)
        :param q_params: the parameters of distributions of z=[z1, z2, ..., zn] for q (approximate posterior)
        :param kwargs:
        :return: loss value
        """
        x_mu = x_hat_params['mu']

        bce = F.binary_cross_entropy(x_mu, x.view(x_mu.size()), reduction='sum')
        kld = 0.
        for i in range(len(p_params)):
            q_mu, q_logvar = q_params[i]['mu'], q_params[i]['logvar']
            p_mu, p_logvar = p_params[i]['mu'], p_params[i]['logvar']
            kld += -0.5 * torch.sum((q_logvar - p_logvar).exp()
                                    + (p_mu - q_mu).pow(2)/(p_logvar.exp())
                                    + p_logvar - q_logvar - 1)

        return dict(loss=bce+kld, bce=bce, kld=kld)
