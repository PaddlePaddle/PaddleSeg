import numpy as np
import PIL
from PIL import Image


def get_new_size(img_size, resize_size):
    if isinstance(resize_size, int) or len(
            resize_size) == 1:  # specified size only for the smallest edge
        w, h = img_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = resize_size if isinstance(resize_size,
                                                        int) else resize_size[0]

        new_short, new_long = requested_new_short, int(requested_new_short *
                                                       long / short)

        new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                             new_short)

    else:  # specified both h and w
        new_w, new_h = resize_size[1], resize_size[0]
    return (new_w, new_h)


class ResizeImage(object):
    """ resize image """

    def __init__(self, resize_size=None, interpolation=Image.BILINEAR):

        self.resize_size = resize_size
        self.interpolation = interpolation

    def __call__(self, img):
        size = get_new_size(img.size, self.resize_size)
        img = img.resize(size, self.interpolation)
        return img


class CenterCropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        return center_crop(img, self.size)


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        img = (img * self.scale - self.mean) / self.std
        return img


class ToCHW(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img


def center_crop(img, size, is_color=True):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(size, (list, tuple)):
        size = size[0]
    h, w = img.shape[:2]
    h_start = (h - size) // 2
    w_start = (w - size) // 2
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        img = img[h_start:h_end, w_start:w_end, :]
    else:
        img = img[h_start:h_end, w_start:w_end]
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
