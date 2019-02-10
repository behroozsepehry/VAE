from torch import nn


class Model(nn.Module):
    def __init__(self, layer_sizes):
        super(Model, self).__init__()
        layers = []
        for l in range(len(layer_sizes)-1):
            layers += [nn.Linear(layer_sizes[l], layer_sizes[l+1]),
                       nn.ReLU(True),]
        self.mlp = nn.Sequential(*layers)

    def forward(self, *input):
        return self.mlp(*input)
