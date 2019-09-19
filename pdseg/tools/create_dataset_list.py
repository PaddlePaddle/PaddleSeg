#!/usr/bin/python
# -*- coding: UTF-8 -*-
import glob
import os.path
import argparse

folder_name = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

postfix = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

data_format = {
    'image': 'png',
    'label': 'png',
}


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg generate file list on cityscapes')
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='dataset root directory',
        default=None,
        type=str)
    parser.add_argument('--file_splitor',
                        dest='file_splitor',
                        help='file list splitor',
                        default=None,
                        type=str)
    return parser.parse_args()


def get_files(image_or_label, dataset_split, args):
    dataset_root = args.dataset_root
    pattern = '*%s.%s' % (postfix[image_or_label], data_format[image_or_label])
    search_files = os.path.join(
        dataset_root, folder_name[image_or_label], dataset_split, '*', pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)


def generate_list(dataset_split, args):
    dataset_root = args.dataset_root
    file_splitor = args.file_splitor

    image_files = get_files('image', dataset_split, args)
    label_files = get_files('label', dataset_split, args)

    num_images = len(image_files)

    file_list = os.path.join(dataset_root, dataset_split + '.list')
    with open(file_list, "w") as f:
        for item in range(num_images):
            left = image_files[item].replace(dataset_root, '')
            if left[0] == os.path.sep:
                left = left.lstrip(os.path.sep)
            right = label_files[item].replace(dataset_root, '')
            if right[0] == os.path.sep:
                right = right.lstrip(os.path.sep)
            line = left + file_splitor + right + '\n'
            f.write(line)


if __name__ == '__main__':
    args = parse_args()
    for dataset_split in ['train', 'val', 'test']:
        generate_list(dataset_split, args)
