import torch
from torch import nn


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ModelBase, self).__init__()
        self.save_path = kwargs.get('save_path')
        self.load_path = kwargs.get('load_path')

    def load(self, path=None, *args, **kwargs):
        if not path:
            path = self.load_path
        if path:
            data = torch.load(path, map_location=kwargs.get('map_location'))
            self.load_state_dict(data['state_dict'])

    def save(self, path=None, *args, **kwargs):
        save_data = kwargs.get('save_data', {})
        assert type(save_data) == dict
        if not path:
            path = self.save_path
        if path:
            torch.save(dict(**save_data, state_dict=self.state_dict()), path)
