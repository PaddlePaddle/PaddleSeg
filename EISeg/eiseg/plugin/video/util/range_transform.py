# The video propagation and fusion code was heavily based on https://github.com/hkchengrex/MiVOS
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/hkchengrex/MiVOS/blob/main/LICENSE

from paddle.vision import transforms

im_mean = (124, 116, 104)

im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225], )

inv_im_trans = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225], )
