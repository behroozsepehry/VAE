from torch import nn

from models.gan import base


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(28 * 28, 400),
                    nn.ReLU(True),
                    nn.Linear(400, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))

        class Generator(nn.Module):
            def __init__(self, z_dim):
                super(Generator, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(z_dim, 400),
                    nn.ReLU(True),
                    nn.Linear(400, 28 * 28),
                    nn.Sigmoid(),
                )

            def forward(self, z):
                return self.fc(z).view(z.size(0), 1, 28, 28)

        self.discriminator = Discriminator()
        self.generator = Generator(self.z_args['z_dim'])
