# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os.path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PaddleSeg generate file list on cityscapes or your customized dataset.')
    parser.add_argument(
        'dataset_root',
        help='dataset root directory',
        type=str
    )
    parser.add_argument(
        '--type',
        help='dataset type: \n'
             '- cityscapes \n'
             '- custom(default)',
        default="custom",
        type=str
    )
    parser.add_argument(
        '--separator',
        dest='separator',
        help='file list separator',
        default="|",
        type=str
    )
    parser.add_argument(
        '--folder',
        help='the folder names of images and labels',
        type=str,
        nargs=2,
        default=['images', 'annotations']
    )
    parser.add_argument(
        '--second_folder',
        help='the second-level folder names of train set, validation set, test set',
        type=str,
        nargs='*',
        default=['train', 'val', 'test']
    )
    parser.add_argument(
        '--format',
        help='data format of images and labels, e.g. jpg or png.',
        type=str,
        nargs=2,
        default=['jpg', 'png']
    )
    parser.add_argument(
        '--postfix',
        help='postfix of images or labels',
        type=str,
        nargs=2,
        default=['', '']
    )

    return parser.parse_args()


def cityscape_cfg(args):
    args.postfix = ['_leftImg8bit', '_gtFine_labelTrainIds']

    args.folder = ['leftImg8bit', 'gtFine']

    args.format = ['png', 'png']


def get_files(image_or_label, dataset_split, args):
    dataset_root = args.dataset_root
    postfix = args.postfix
    format = args.format
    folder = args.folder

    pattern = '*%s.%s' % (postfix[image_or_label], format[image_or_label])

    search_files = os.path.join(dataset_root, folder[image_or_label],
                                dataset_split, pattern)
    search_files2 = os.path.join(dataset_root, folder[image_or_label],
                                 dataset_split, "*", pattern)  # 包含子目录
    search_files3 = os.path.join(dataset_root, folder[image_or_label],
                                 dataset_split, "*", "*", pattern)  # 包含三级目录

    filenames = glob.glob(search_files)
    filenames2 = glob.glob(search_files2)
    filenames3 = glob.glob(search_files3)

    filenames = filenames + filenames2 + filenames3

    return sorted(filenames)


def generate_list(args):
    dataset_root = args.dataset_root
    separator = args.separator

    for dataset_split in args.second_folder:
        print("Creating {}.txt...".format(dataset_split))
        image_files = get_files(0, dataset_split, args)
        label_files = get_files(1, dataset_split, args)
        if not image_files:
            img_dir = os.path.join(dataset_root, args.folder[0], dataset_split)
            num_images = 0
            print("No files in {}".format(img_dir))
        else:
            num_images = len(image_files)

        if not label_files:
            label_dir = os.path.join(dataset_root, args.folder[1], dataset_split)
            num_label = 0
            print("No files in {}".format(label_dir))
        else:
            num_label = len(label_files)

        if num_images == num_label:
            num = num_images
        else:
            print("number of images = {} and number of labels = {} are not equal."
                  .format(num_images, num_label))
            num = max(num_images, num_label)

        file_list = os.path.join(dataset_root, dataset_split + '.txt')
        with open(file_list, "w") as f:
            for item in range(num):
                try:
                    left = image_files[item].replace(dataset_root, '')
                    if left[0] == os.path.sep:
                        left = left.lstrip(os.path.sep)
                except:
                    left = ''

                try:
                    right = label_files[item].replace(dataset_root, '')
                    if right[0] == os.path.sep:
                        right = right.lstrip(os.path.sep)
                except:
                    right = ''

                line = left + separator + right + '\n'
                f.write(line)
                print(line)


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'cityscapes':
        cityscape_cfg(args)
    generate_list(args)
