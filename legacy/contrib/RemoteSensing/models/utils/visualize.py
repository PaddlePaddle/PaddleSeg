import os
import os.path as osp
import numpy as np
from PIL import Image as Image


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map


def splice_imgs(img_list, vis_path):
    """Splice pictures horizontally
    """
    IMAGE_WIDTH, IMAGE_HEIGHT = img_list[0].size
    padding_width = 20
    img_num = len(img_list)
    to_image = Image.new('RGB',
                         (img_num * IMAGE_WIDTH + (img_num - 1) * padding_width,
                          IMAGE_HEIGHT))  # Create a new picture
    padding = Image.new('RGB', (padding_width, IMAGE_HEIGHT), (255, 255, 255))

    # Loop through, paste each picture to the corresponding position in order
    for i, from_image in enumerate(img_list):
        to_image.paste(from_image, (i * (IMAGE_WIDTH + padding_width), 0))
        if i < img_num - 1:
            to_image.paste(padding,
                           (i * (IMAGE_WIDTH + padding_width) + IMAGE_WIDTH, 0))
    return to_image.save(vis_path)
