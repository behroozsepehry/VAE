from models.vae import base_normal_sparse, cnn_28_normal


class Model(base_normal_sparse.NormalSparseBase, cnn_28_normal.Model):
    pass
