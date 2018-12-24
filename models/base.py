import torch
from torch import nn


class VaeModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VaeModelBase, self).__init__()
        self.save_path = kwargs.get('save_path')
        self.load_path = kwargs.get('load_path')
        self.load(self.load_path)

    def load(self, path, *args, **kwargs):
        if path:
            data = torch.load(path)
            self.load_state_dict(data['state_dict'])

    def save(self, path, *args, **kwargs):
        save_data = kwargs.get('save_data', {})
        assert type(save_data) == dict
        if path:
            torch.save(dict(**save_data, state_dict=self.state_dict()), path)

    def encode(self, *args, **kwargs):
        # return tuple
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        # return tuple
        # the first output must be reconstruction
        raise NotImplementedError

    def reparameterize(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        assert len(args) == 1
        x, = tuple(args)
        z_params = self.encode(x)
        z = self.reparameterize(*z_params)
        x_params = self.decode(z)
        return x_params + z_params
