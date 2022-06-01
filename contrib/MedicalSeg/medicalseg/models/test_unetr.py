
import paddle as torch
from unetr import UNETR

def test_unettr():

    model = UNETR(img_shape=(64, 64, 64), input_dim=1, output_dim=1)
    a = torch.rand(shape=[1, 1, 64, 64, 64])


    assert model(a).shape == [1,1,64,64,64]
    del model
    model = UNETR(img_shape=(64, 64, 64), input_dim=1, output_dim=1)
    assert model(a).shape == [1, 1, 64, 64, 64]

    num_heads = 12  # 12 normally
    embed_dim = 768  # 768 normally
    roi_size = (128,128,64)
    model = UNETR(img_shape=tuple(roi_size), input_dim=4, output_dim=3,
                  embed_dim=embed_dim, patch_size=16, num_heads=num_heads,
                  ext_layers=[3, 6, 9, 12], norm='instance',
                  base_filters=16,
                  dim_linear_block=3072)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    print('different parameters from the official model:', pytorch_total_params - 101910630)

test_unettr()