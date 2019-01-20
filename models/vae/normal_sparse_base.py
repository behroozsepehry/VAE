import torch
import torch.nn as nn

from models.vae import normal_base


class NormalSparseBase(normal_base.VaeModelNormalBase):
    def __init__(self, *args, **kwargs):
        super(NormalSparseBase, self).__init__(*args, **kwargs)
        z2_dim = self.z_args['z2_dim']
        z_dim = self.z_args['z_dim']
        self.fc_d_1 = nn.Linear(z_dim, z2_dim)
        self.fc_d_2 = nn.Sequential(nn.Linear(z2_dim, z_dim),
                                    nn.ReLU(True),
                                    )

    def _denoise(self, z, **kwargs):
        h1 = self.fc_d_1(z)
        if self.training:
            z_2 = torch.sigmoid(h1)
        else:
            z_2 = (torch.sign(h1) + 1.)/2.
        z_d = self.fc_d_2(z_2)
        return dict(z_d=z_d, z_2=z_2)

    def decode(self, z, **kwargs):
        denoised = self._denoise(z)
        z_d, z_2 = denoised['z_d'], denoised['z_2']
        x_params = self._decode(z_d)
        return dict(**x_params, z_2=z_2)

