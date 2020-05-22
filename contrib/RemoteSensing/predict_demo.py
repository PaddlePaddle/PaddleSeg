# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
from PIL import Image as Image
import argparse
from models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing predict')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--load_model_dir',
        dest='load_model_dir',
        help='model load directory',
        default=None,
        type=str)
    return parser.parse_args()


args = parse_args()

data_dir = args.data_dir
load_model_dir = args.load_model_dir

# predict
model = load_model(load_model_dir)
pred_dir = osp.join(load_model_dir, 'predict')
if not osp.exists(pred_dir):
    os.mkdir(pred_dir)

val_list = osp.join(data_dir, 'val.txt')
color_map = [0, 0, 0, 255, 255, 255]
with open(val_list) as f:
    lines = f.readlines()
    for line in lines:
        img_path = line.split(' ')[0]
        print('Predicting {}'.format(img_path))
        img_path_ = osp.join(data_dir, img_path)

        pred = model.predict(img_path_)

        # 以伪彩色png图片保存预测结果
        pred_name = osp.basename(img_path).rstrip('npy') + 'png'
        pred_path = osp.join(pred_dir, pred_name)
        pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
        pred_mask.save(pred_path)
