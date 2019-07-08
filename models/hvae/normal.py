from typing import List, Dict, Tuple, Any
import torch

import models.base
from utilities import general_utilities as g_util
from utilities import main_utilities as m_util


class Model(models.base.ModelBase):
    def __init__(self,
                 enf_args: Dict[str, Any],
                 def_args: Dict[str, Any],
                 enz_args_list: List[Dict[str, Any]],
                 dez_args_list: List[Dict[str, Any]],
                 *args, **kwargs):
        """
        :param enf_args: encoder feature extractor model args
        :param def_args: decoder feature extractor model args
        :param enz_args_list: encoder z constructor models args (each hierarchy of z has one arg)
        :param dez_args_list: decoder z constructor models args (each hierarchy of z has one arg)
        :param args:
        :param kwargs:
        """
        super(Model, self).__init__(*args, **kwargs)
        self.enf = m_util.get_model(**enf_args)
        self.def_ = m_util.get_model(**def_args)
        self.enz = [m_util.get_model(**enz_args) for enz_args in enz_args_list]
        self.dez = [m_util.get_model(**dez_args) for dez_args in dez_args_list]

    def get_z_args(self, zz: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mu, z_logvar = torch.split(zz, zz.size(1) // 2, dim=1)
        return dict(z_mu=z_mu, z_logvar=z_logvar)

    def encode(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        features = self.enf(x)
        z_params = []
        for i in range(len(self.enz)):
            encoder = self.enz[i]
            zz = encoder(features)
            z_params = self.get_z_args(zz)
            z = self.reparameterize(**z_params)
            z_params['z'] = z
            z_params.append(z_params)
            features = torch.cat((features, z))
        return z_params

    def get_total_z(self, z_params: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.cat([z_p['z'] for z_p in z_params], dim=1)

    def decode(self, z_params: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        z = self.get_total_z(z_params)


    def generate_z(self):
        pass

    def reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5*z_logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(z_mu)
        return z

    def forward(self, x, **kwargs):
        z_params = self.encode(x)
        z_r = self.reparameterize(**z_params)
        x_params = self.decode(z_r['z'])
        return dict(x=x, **x_params, **z_params, **z_r)

    def forward_backward(self, x, loss_functions, optimizers, **kwargs):
        model_out = self(x)
        train_batch_losses = {}
        for k in loss_functions:
            if k != m_util.RECURSION_CHAR:
                optimizers[k].zero_grad()
                loss_vals = loss_functions[k](**model_out)
                loss_vals['loss'].backward()
                train_batch_losses.update(g_util.append_key_dict(loss_vals, k+'_'))
                optimizers[k].step()
            else:
                inner_model_losses = self.inner_model.forward_backward(model_out['z'].detach(),
                                                  loss_functions[m_util.RECURSION_CHAR],
                                                  optimizers[m_util.RECURSION_CHAR],
                                                  **kwargs.get(m_util.RECURSION_CHAR, {}))
                train_batch_losses[m_util.RECURSION_CHAR] = inner_model_losses
        return dict(losses=train_batch_losses)

    def get_parameters_groups(self):
        p_group = {'vae': self.parameters()}
        if self.inner_model:
            p_group[m_util.RECURSION_CHAR] = self.inner_model.get_parameters_groups()
        return p_group
