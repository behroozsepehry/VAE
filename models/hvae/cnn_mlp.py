from typing import List, Dict
import torch

import models.base
from utilities import general_utilities as g_util
from utilities import main_utilities as m_util


class Model(models.base.ModelBase):
    def __init__(self, conv_args: Dict, deconv_args: Dict, mlp_args_list: List[Dict], *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.conv = m_util.get_model(**conv_args)
        self.deconv = m_util.get_model(**deconv_args)
        self.mlp_list = [m_util.get_model(**mlp_args) for mlp_args in mlp_args_list]

    def encode(self, x):
        features = self.conv(x)
        z_params = []
        for i in range(len(self.mlp_list)):
            zz = self.mlp_list[i](features)
            z_mu, z_logvar = torch.split(zz, zz.size(1) // 2, dim=1)
            z = self.reparameterize(z_mu, z_logvar)
            z_params.append({'z': z, 'z_mu': z_mu, 'z_logvar': z_logvar})
            features = torch.cat((features, z))
        return z_params

    def decode(self, z_params, **kwargs):
        return self._decode(z, **kwargs)

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
