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
To visualize the annotation:
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
    parser.add_argument('--pred_dir',
                        help='the dir of predicted images',
                        type=str)
    parser.add_argument('--save_dir',
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


def visualize_origin_annotated_imgs(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    images_path = get_images_path(args.file_path)
    bar = progbar.Progbar(target=len(images_path), verbose=1)

    for idx, (origin_path, annot_path) in enumerate(images_path):
        origin_img = Image.open(origin_path)
        annot_img = Image.open(annot_path)
        annot_img = np.array(annot_img)

        bar.update(idx + 1)
        if len(np.unique(annot_img)) == 1:
            continue

        # weighted image
        color_map = visualize.get_color_map_list(256)
        weighted_img = utils.visualize.visualize(origin_path,
                                                 annot_img,
                                                 color_map,
                                                 weight=0.6)
        weighted_img = Image.fromarray(
            cv2.cvtColor(weighted_img, cv2.COLOR_BGR2RGB))

        # result image
        result_img = visualize.paste_images([origin_img, weighted_img])

        # save
        image_name = os.path.split(origin_path)[-1]
        result_img.save(os.path.join(args.save_dir, image_name))


def analyze_annot(annot_img, ratio_thr=0.1):
    """
    Analyze the annotated image.
    """
    if not isinstance(annot_img, np.ndarray):
        annot_img = np.array(annot_img)
    assert annot_img.ndim == 2

    if len(np.unique(annot_img)) == 1:
        annot_ratio = 0.
    else:
        annot_nums = np.bincount(annot_img.flatten())
        assert len(annot_nums) == 2
        annot_ratio = annot_nums / sum(annot_nums)
        annot_ratio = annot_ratio[1]

    res = annot_ratio > ratio_thr
    return res, annot_ratio


def analyze_pred(ori_img,
                 pred_img,
                 percent_ratio=3,
                 ratio_thr=0.1,
                 max_val_thr=252,
                 min_val_thr=30):
    """
    Analyze the predicted image.
    """
    if not isinstance(ori_img, np.ndarray):
        ori_img = np.array(ori_img)
    if not isinstance(pred_img, np.ndarray):
        pred_img = np.array(pred_img)
    assert ori_img.ndim == 3 and pred_img.ndim == 2

    # analyze ratio
    if len(np.unique(pred_img)) == 1:
        pred_ratio = 0.
    else:
        pred_nums = np.bincount(pred_img.ravel())
        assert len(pred_nums) == 2
        pred_ratio = pred_nums / sum(pred_nums)
        pred_ratio = pred_ratio[1]

    # analyze Value in HSV
    v_img = np.max(ori_img, 2)  # get Value
    jishui_pixel = v_img[pred_img == 1]
    if jishui_pixel.size > 0:
        min_val = np.percentile(jishui_pixel, percent_ratio)
        max_val = np.percentile(jishui_pixel, 100 - percent_ratio)
    else:
        min_val = max_val = 0

    # analyze
    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)
    h_img = hsv_img[:, :, 0]
    s_img = hsv_img[:, :, 1]
    v_img = hsv_img[:, :, 2]
    h_mean_val = np.mean(h_img)
    s_mean_val = np.mean(s_img)
    v_mean_val = np.mean(v_img)

    res = pred_ratio > ratio_thr
    '''
    if (pred_ratio < ratio_thr) or (max_val > max_val_thr):
        res = False
    else:
        res = True
    '''

    return res, pred_ratio, min_val, max_val, h_mean_val, s_mean_val, v_mean_val


def visualize_origin_annot_pred_imgs(args):
    save_dir = args.save_dir
    pred_dir = args.pred_dir
    file_path = args.file_path

    ratio_thr = 0.1
    cls_dict = {
        (True, True): 'tp',
        (True, False): 'fn',
        (False, True): 'fp',
        (False, False): 'tn'
    }
    cls_res = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    images_path = get_images_path(file_path)
    bar = progbar.Progbar(target=len(images_path), verbose=1)

    mkdir(save_dir)
    mkdir(os.path.join(save_dir, 'ratio_thr_' + str(ratio_thr) + '_tp'), True)
    mkdir(os.path.join(save_dir, 'ratio_thr_' + str(ratio_thr) + '_fp'), True)
    mkdir(os.path.join(save_dir, 'ratio_thr_' + str(ratio_thr) + '_tn'), True)
    mkdir(os.path.join(save_dir, 'ratio_thr_' + str(ratio_thr) + '_fn'), True)

    for idx, (origin_path, annot_path) in enumerate(images_path):
        origin_img = Image.open(origin_path)
        annot_img = Image.open(annot_path)
        annot_img = np.array(annot_img)

        # weighted annotated image
        color_map = visualize.get_color_map_list(256)
        wt_annot_img = utils.visualize.visualize(origin_path,
                                                 annot_img,
                                                 color_map,
                                                 weight=0.6)
        wt_annot_img = Image.fromarray(
            cv2.cvtColor(wt_annot_img, cv2.COLOR_BGR2RGB))

        # weighted predicted image
        image_name = os.path.split(origin_path)[-1]
        tmp_name = image_name.replace('jpg', 'png')
        pred_path = os.path.join(pred_dir, tmp_name)
        if not os.path.exists(pred_path):
            print('{} is not existed'.format(pred_path))
            continue
        pred_img = np.array(Image.open(pred_path))
        wt_pred_img = utils.visualize.visualize(origin_path,
                                                pred_img,
                                                color_map,
                                                weight=0.6)
        wt_pred_img = Image.fromarray(
            cv2.cvtColor(wt_pred_img, cv2.COLOR_BGR2RGB))

        # analyze
        annot_res, annot_ratio = analyze_annot(annot_img, ratio_thr)
        pred_res, pred_ratio, pred_min, pred_max, hm, sm, vm = analyze_pred(
            origin_img, pred_img, ratio_thr=ratio_thr)
        cls_name = cls_dict[(annot_res, pred_res)]
        cls_res[cls_name] += 1

        # result image
        result_img = visualize.paste_images(
            [origin_img, wt_annot_img, wt_pred_img])
        result_img = result_img.convert('RGB')
        image_name = image_name.split(".")[0]
        image_name = "{}_{:.3f}_{:.3f}_min{}_max{}_{:.3f}_{:.3f}_{:.3f}.jpg".format(
            image_name, annot_ratio, pred_ratio, pred_min, pred_max, hm, sm, vm)
        result_img.save(
            os.path.join(save_dir,
                         'ratio_thr_' + str(ratio_thr) + '_' + cls_name,
                         image_name))
        bar.update(idx + 1)

    precision = cls_res['tp'] / (cls_res['tp'] + cls_res['fp'])
    recall = cls_res['tp'] / (cls_res['tp'] + cls_res['fn'])
    print(cls_res)
    print("precision: {:.3f}, recall: {:.3f}".format(precision, recall))


if __name__ == '__main__':
    args = parse_args()
    visualize_origin_annot_pred_imgs(args)
