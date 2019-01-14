import torch


class Sampler(object):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', 'standard_normal')
        self.device = kwargs.get('device')
        if self.name == 'standard_normal':
            self._sampler_func = lambda size: torch.randn(*size, device=self.device)
        elif self.name == 'uniform_int':
            self.low = kwargs.get('low', 0)
            self.high = kwargs.get('high', 2)
            self._sampler_func = lambda size: torch.randint(self.low, self.high, size,
                                                            device=self.device, dtype=torch.float)
        else:
            raise NotImplementedError

    def __call__(self, size):
        return self._sampler_func(size)
