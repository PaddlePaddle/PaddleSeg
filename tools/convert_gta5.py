import os
import argparse
import numpy as np
import glob
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA5 label into 19 classes')
    parser.add_argument(
        "--dataset_root",
        dest="dataset_root",
        default='../data/GTA5/labels',
        type=str)
    return parser.parse_args()


def main(args):
    label_colours = [
        128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
        153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152,
        0, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0,
        80, 100, 0, 0, 230, 119, 11, 32
    ]

    label_colours = label_colours + (255 - 19) * [0, 0, 0]
    print(label_colours)
    ignore_label = 255
    id_to_trainid = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18
    }
    labellist = glob.glob(os.path.join(args.dataset_root, '*.png'))
    assert len(labellist) == 24966, print(
        'There is only {} labels, we actually need 24966 labels')

    print('start converting...')
    count = 0
    for labelid in labellist:
        label = np.array(Image.open(labelid), np.float32)
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            binary_mask = (label == k)
            label_copy[binary_mask] = v
        assert len(np.unique(label_copy)) <= 20, print(
            'The image have more than 20 classes after converting')
        new_label = Image.fromarray(label_copy.astype(np.uint8)).convert('P')
        new_label.putpalette(label_colours)
        savepath = labelid.replace('.png', 'color19.png')
        new_label.save(savepath)
        count += 1
        if count % 10 == 0:
            print('processed {} images'.format(count))


if __name__ == '__main__':
    args = parse_args()
    main(args)
