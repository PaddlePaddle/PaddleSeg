# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
from pdseg.vis import get_color_map_list


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir_or_file',
                        help='input gray label directory or file list path')
    parser.add_argument('output_dir',
                        help='output colorful label directory')
    parser.add_argument('--dataset_dir',
                        help='dataset directory')
    parser.add_argument('--file_separator',
                        help='file list separator')
    return parser.parse_args()


def gray2pseudo_color(args):
    """将灰度标注图片转换为伪彩色图片"""
    input = args.dir_or_file
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating colorful label directory:', output_dir)

    color_map = get_color_map_list(256)
    if os.path.isdir(input):
        for grt_path in glob.glob(osp.join(input, '*.png')):
            print('Converting original label:', grt_path)
            basename = osp.basename(grt_path)

            im = Image.open(grt_path)
            lbl = np.asarray(im)

            lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
            lbl_pil.putpalette(color_map)

            new_file = osp.join(output_dir, basename)
            lbl_pil.save(new_file)
    elif os.path.isfile(input):
        if args.dataset_dir is None or args.file_separator is None:
            print('No dataset_dir or file_separator input!')
            sys.exit()
        with open(input) as f:
            for line in f:
                parts = line.strip().split(args.file_separator)
                grt_name = parts[1]
                grt_path = os.path.join(args.dataset_dir, grt_name)

                print('Converting original label:', grt_path)
                basename = osp.basename(grt_path)

                im = Image.open(grt_path)
                lbl = np.asarray(im)

                lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)

                new_file = osp.join(output_dir, basename)
                lbl_pil.save(new_file)
    else:
        print('It\'s neither a dir nor a file')


if __name__ == '__main__':
    args = parse_args()
    gray2pseudo_color(args)
