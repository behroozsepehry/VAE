from torch import nn


class Model(nn.Module):
    def __init__(self, layer_sizes, **kwargs):
        super(Model, self).__init__()
        layers = []
        for l in range(len(layer_sizes)-1):
            layers += [nn.Linear(layer_sizes[l], layer_sizes[l+1]),
                       nn.ReLU(True),]

        activation_name = kwargs.get('activation')
        if activation_name:
            del layers[-1]
            layers.append(getattr(nn, activation_name)())
        self.mlp = nn.Sequential(*layers)

    def forward(self, *input):
        return self.mlp(*input)
