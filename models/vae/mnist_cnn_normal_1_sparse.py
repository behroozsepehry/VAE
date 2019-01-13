from models.vae import normal_sparse_base, mnist_cnn_normal_1


class Model(normal_sparse_base.NormalSparseBase, mnist_cnn_normal_1.Model):
    def decode(self, z, **kwargs):
        denoised = self._denoise(z)
        z_d, z_2 = denoised['z_d'], denoised['z_2']
        x_params = super(Model, self).decode(z_d)
        return dict(**x_params, z_2=z_2)
