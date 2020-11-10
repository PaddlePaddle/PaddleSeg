import paddleseg.datasets.cityscapes_labels as cl
import os
import glob
import numpy as np
from PIL import Image

# from paddleseg.utils.visualize import get_color_map_list


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


dataset_root = '/ssd1/home/chulutao/dataset/cityscapes-for-nvidia'

label_dir = os.path.join(dataset_root, 'autolabelled', 'train_extra')
if dataset_root is None or not os.path.isdir(dataset_root) or not os.path.isdir(
        label_dir):
    raise ValueError(
        "The dataset is not Found or the folder structure is nonconfoumance.")

autolabeling_label_files = sorted(
    glob.glob(os.path.join(label_dir, '*', '*_leftImg8bit.png')))

for file in autolabeling_label_files:

    mask = np.array(Image.open(file))
    for k, v in cl.label2trainid.items():
        # print(k, v)
        binary_mask = (mask == k)
        mask[binary_mask] = v
    # print(np.unique(mask))
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(get_color_map_list(255))
    aaa = file.split('/')
    path = os.path.join(dataset_root, 'convert_autolabelled', aaa[-2])
    if not os.path.exists(path):
        os.makedirs(path)
    new_mask.save(os.path.join(path, aaa[-1]))
