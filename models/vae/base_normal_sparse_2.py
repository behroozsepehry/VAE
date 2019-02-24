import torch

from models.vae import base_normal_sparse


class NormalSparseBase2(base_normal_sparse.NormalSparseBase):
    def __init__(self, *args, **kwargs):
        super(NormalSparseBase2, self).__init__(*args, **kwargs)

    def _denoise(self, z, **kwargs):
        h1 = self.fcd_1(z)
        if self.training:
            z_2 = h1
        else:
            z_2 = torch.where(h1.abs() > self.z_args['threshold'], h1, torch.tensor([0.], device=h1.device))
        z_d = self.fcd_2(z_2)
        return dict(z_d=z_d, z_2=z_2)

