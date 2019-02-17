from losses import base

class Loss(base.LossBase):
    def __init__(self, generator, discriminator, dataloader):
