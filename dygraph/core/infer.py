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

from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle.fluid as fluid
import cv2
import tqdm

import utils
import utils.logging as logging


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def infer(model, test_dataset=None, model_dir=None, save_dir='output'):
    ckpt_path = os.path.join(model_dir, 'model')
    para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
    model.set_dict(para_state_dict)
    model.eval()

    added_saved_dir = os.path.join(save_dir, 'added')
    pred_saved_dir = os.path.join(save_dir, 'prediction')

    logging.info("Start to predict...")
    for im, im_info, im_path in tqdm.tqdm(test_dataset):
        im = to_variable(im)
        pred, _ = model(im)
        pred = pred.numpy()
        pred = np.squeeze(pred).astype('uint8')
        for info in im_info[::-1]:
            if info[0] == 'resize':
                h, w = info[1][0], info[1][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            elif info[0] == 'padding':
                h, w = info[1][0], info[1][1]
                pred = pred[0:h, 0:w]
            else:
                raise Exception("Unexpected info '{}' in im_info".format(
                    info[0]))

        im_file = im_path.replace(test_dataset.data_dir, '')
        if im_file[0] == '/':
            im_file = im_file[1:]
        # save added image
        added_image = utils.visualize(im_path, pred, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save prediction
        pred_im = utils.visualize(im_path, pred, weight=0.0)
        pred_saved_path = os.path.join(pred_saved_dir, im_file)
        mkdir(pred_saved_path)
        cv2.imwrite(pred_saved_path, pred_im)
