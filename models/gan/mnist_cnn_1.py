from torch import nn

from models.gan import base


class Model(base.GanModelBase):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, 3, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1),
                    nn.Conv2d(16, 32, 3, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1),
                    nn.Conv2d(32, 32, 3, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1),
                    nn.Conv2d(32, 32, 3, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1),
                    nn.Conv2d(32, 16, 3, stride=1, padding=0),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, stride=1),
                )
                self.fc = nn.Sequential(
                    nn.Linear(16 * 13 * 13, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                h = self.conv(x)
                c = self.fc(h.view(h.size(0), -1))
                return c

        class Generator(nn.Module):
            def __init__(self, z_dim):
                super(Generator, self).__init__()
                self.fc = nn.Sequential(nn.Linear(z_dim, 7 * 7 * 32),
                                        nn.ReLU(True),
                                        )
                self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(32, 32, 3, stride=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 3, stride=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 3, stride=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 16, 3, stride=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(16, 1, 4, stride=2, padding=2),
                    nn.Sigmoid(),
                )

            def forward(self, z):
                h = self.fc(z).view(z.size(0), 32, 7, 7)
                x = self.deconv(h)
                return x

        self.discriminator = Discriminator()
        self.generator = Generator(self.z_dim)
