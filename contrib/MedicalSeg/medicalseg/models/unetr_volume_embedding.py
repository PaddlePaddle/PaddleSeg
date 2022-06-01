
import paddle as torch
import paddle
import paddle.nn as nn


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return  paddle.tile(tensor, repeat_times=(tile,1,1))



class AbsPositionalEncoding1D(nn.Layer):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        params=torch.randn(shape=[1,tokens, dim])
        self.abs_pos_enc = paddle.create_parameter(shape=params.shape,
                        dtype=str(params.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(params))
    def forward(self, x):
        batch = x.shape[0]
        expb=expand_to_batch(self.abs_pos_enc, desired_size=batch)
        return x + expb
        
class Embeddings3D(nn.Layer):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size=16, dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3D(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size, bias_attr=False)
        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x is a 5D tensor
        """
        patch_embeddings=self.patch_embeddings(x)
        shape =patch_embeddings.shape
        _d=paddle.reshape(patch_embeddings,[shape[0],shape[1],-1])
        _d=paddle.transpose(_d,perm=[0,2,1])
        embeddings = self.dropout(self.position_embeddings(_d))
        return embeddings