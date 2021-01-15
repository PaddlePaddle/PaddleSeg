# coding: utf8
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

import os
import os.path as osp
import sys
import numpy as np
from PIL import Image as Image
import argparse
from models import load_model
from models.utils.visualize import get_color_map_list
from utils import paddle_utils


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing predict')
    parser.add_argument(
        '--single_img',
        dest='single_img',
        help='single image path to predict',
        default=None,
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--file_list',
        dest='file_list',
        help='file name of predict file list',
        default=None,
        type=str)
    parser.add_argument(
        '--load_model_dir',
        dest='load_model_dir',
        help='model load directory',
        default=None,
        type=str)
    parser.add_argument(
        '--save_img_dir',
        dest='save_img_dir',
        help='save directory name of predict results',
        default='predict_results',
        type=str)
    parser.add_argument(
        '--color_map',
        dest='color_map',
        help='color map of predict results',
        type=int,
        nargs='*',
        default=-1)
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


paddle_utils.enable_static()
args = parse_args()
data_dir = args.data_dir
file_list = args.file_list
single_img = args.single_img
load_model_dir = args.load_model_dir
save_img_dir = args.save_img_dir
if not osp.exists(save_img_dir):
    os.makedirs(save_img_dir)
if args.color_map == -1:
    color_map = get_color_map_list(256)
else:
    color_map = args.color_map

# predict
model = load_model(load_model_dir)

if single_img is not None:
    pred = model.predict(single_img)
    # 以伪彩色png图片保存预测结果
    pred_name, _ = osp.splitext(osp.basename(single_img))
    pred_path = osp.join(save_img_dir, pred_name + '.png')
    pred_mask = Image.fromarray(pred['label_map'].astype(np.uint8), mode='P')
    pred_mask.putpalette(color_map)
    pred_mask.save(pred_path)
    print('Predict result is saved in {}'.format(pred_path))
elif (file_list is not None) and (data_dir is not None):
    with open(osp.join(data_dir, file_list)) as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.split(' ')[0]
            img_path_ = osp.join(data_dir, img_path)

            pred = model.predict(img_path_)

            # 以伪彩色png图片保存预测结果
            pred_name, _ = osp.splitext(osp.basename(img_path))
            pred_path = osp.join(save_img_dir, pred_name + '.png')
            pred_mask = Image.fromarray(
                pred['label_map'].astype(np.uint8), mode='P')
            pred_mask.putpalette(color_map)
            pred_mask.save(pred_path)
            print('Predict result is saved in {}'.format(pred_path))
else:
    raise Exception(
        'You should either set the parameter single_img, or set the parameters data_dir and file_list.'
    )
