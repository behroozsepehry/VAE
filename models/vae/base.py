import models.base
from utilities import general_utilities as g_util
from utilities import main_utilities as m_util


class VaeModelBase(models.base.ModelBase):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__(*args, **kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def _decode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z, **kwargs):
        """separate the _decode and decode allows for modified decode that
            calls _decode, e.g., adding sparsity as part of decoding
        """
        return self._decode(z, **kwargs)

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

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
                inner_model_losses = self.inner_model.forward_backward(model_out['z'],
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
