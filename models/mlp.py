from torch import nn


class Model(nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        super(Model, self).__init__()
        n_l = len(layer_sizes) - 1
        assert n_l >= 1
        layers = []
        for l in range(n_l-1):
            layers += [nn.Linear(layer_sizes[l], layer_sizes[l+1]),
                       nn.ReLU(True)]
        layers += [nn.Linear(layer_sizes[n_l-1], layer_sizes[n_l])]
        activation_name = kwargs.get('activation')
        if activation_name:
            layers += [(getattr(nn, activation_name)())]
        self.mlp = nn.Sequential(*layers)

    def forward(self, *input):
        return self.mlp(*input)
