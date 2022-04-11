# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
visualize the annotated images:
1. Add the origin image and annotated image to produce the weighted image.
2. Paste the origin image and weighted image to generate the final image.
"""

import argparse
import os
import sys
import shutil

import cv2
import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddleseg import utils
from paddleseg.utils import logger, progbar, visualize


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--file_path',
        help='The file contains the path of origin and annotated images',
        type=str)
    parser.add_argument(
        '--pred_dir', help='the dir of predicted images', type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the visualized images',
        type=str,
        default='./output/visualize_annotation')
    return parser.parse_args()


def get_images_path(file_path):
    """
    Get the path of origin images and annotated images.
    """
    assert os.path.isfile(file_path)

    images_path = []
    image_dir = os.path.dirname(file_path)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            origin_path, annot_path = line.split(" ")
            origin_path = os.path.join(image_dir, origin_path)
            annot_path = os.path.join(image_dir, annot_path)
            images_path.append([origin_path, annot_path])

    return images_path


def mkdir(dir, rm_exist=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        if rm_exist:
            shutil.rmtree(dir)
            os.makedirs(dir)


def visualize_imgs(args):
    file_path = args.file_path
    pred_dir = args.pred_dir
    save_dir = args.save_dir
    weight = 0.1

    images_path = get_images_path(file_path)
    bar = progbar.Progbar(target=len(images_path), verbose=1)
    mkdir(save_dir, True)

    for idx, (origin_path, annot_path) in enumerate(images_path):
        origin_img = Image.open(origin_path)
        annot_img = Image.open(annot_path)
        annot_img = np.array(annot_img)
        wt_annot_img = None
        wt_pred_img = None
        color_map = visualize.get_color_map_list(256)

        # weighted annoted image
        wt_annot_img = utils.visualize.visualize(
            origin_path, annot_img, color_map, weight=weight)
        wt_annot_img = Image.fromarray(
            cv2.cvtColor(wt_annot_img, cv2.COLOR_BGR2RGB))

        # weighted predicted image
        if pred_dir is not None:
            image_name = os.path.split(origin_path)[-1]
            tmp_name = image_name.replace('jpg', 'png')
            pred_path = os.path.join(pred_dir, tmp_name)
            if os.path.exists(pred_path):
                pred_img = np.array(Image.open(pred_path))
                wt_pred_img = utils.visualize.visualize(
                    origin_path, pred_img, color_map, weight=weight)
                wt_pred_img = Image.fromarray(
                    cv2.cvtColor(wt_pred_img, cv2.COLOR_BGR2RGB))

        # result image
        imgs = [origin_img, wt_annot_img]
        if wt_pred_img is not None:
            imgs.append(wt_pred_img)
        result_img = visualize.paste_images(imgs)

        image_name = os.path.split(origin_path)[-1]
        result_img.save(os.path.join(save_dir, image_name))

        bar.update(idx + 1)


if __name__ == '__main__':
    args = parse_args()
    visualize_imgs(args)
