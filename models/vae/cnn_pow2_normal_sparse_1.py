from models.vae import base_normal_sparse, cnn_pow2_normal


class Model(base_normal_sparse.NormalSparseBase, cnn_pow2_normal.Model):
    pass
