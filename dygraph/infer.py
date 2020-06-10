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

import argparse
import os

from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle.fluid as fluid
import cv2
import tqdm

import transforms as T
import models
import utils
import utils.logging as logging
from utils import get_environ_info


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument('--model_name',
                        dest='model_name',
                        help="Model type for traing, which is one of ('UNet')",
                        type=str,
                        default='UNet')

    # params of dataset
    parser.add_argument('--data_dir',
                        dest='data_dir',
                        help='The root directory of dataset',
                        type=str)
    parser.add_argument('--test_list',
                        dest='test_list',
                        help='Val list file of dataset',
                        type=str,
                        default=None)
    parser.add_argument('--num_classes',
                        dest='num_classes',
                        help='Number of classes',
                        type=int,
                        default=2)

    # params of prediction
    parser.add_argument("--input_size",
                        dest="input_size",
                        help="The image size for net inputs.",
                        nargs=2,
                        default=[512, 512],
                        type=int)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='Mini batch size',
                        type=int,
                        default=2)
    parser.add_argument('--model_dir',
                        dest='model_dir',
                        help='The path of model for evaluation',
                        type=str,
                        default=None)
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        help='The directory for saving the inference results',
                        type=str,
                        default='./output/result')

    return parser.parse_args()


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def infer(model, data_dir=None, test_list=None, model_dir=None,
          transforms=None):
    ckpt_path = os.path.join(model_dir, 'model')
    para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
    model.set_dict(para_state_dict)
    model.eval()

    added_saved_dir = os.path.join(args.save_dir, 'added')
    pred_saved_dir = os.path.join(args.save_dir, 'prediction')

    logging.info("Start to predict...")
    with open(test_list, 'r') as f:
        files = f.readlines()
    for file in tqdm.tqdm(files):
        file = file.strip()
        im_file = os.path.join(data_dir, file)
        im, im_info, _ = transforms(im_file)
        im = np.expand_dims(im, axis=0)
        im = to_variable(im)

        pred, _ = model(im, mode='test')
        pred = pred.numpy()
        pred = np.squeeze(pred).astype('uint8')
        keys = list(im_info.keys())
        for k in keys[::-1]:
            if k == 'shape_before_resize':
                h, w = im_info[k][0], im_info[k][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            elif k == 'shape_before_padding':
                h, w = im_info[k][0], im_info[k][1]
                pred = pred[0:h, 0:w]

        # save added image
        added_image = utils.visualize(im_file, pred, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save prediction
        pred_im = utils.visualize(im_file, pred, weight=0.0)
        pred_saved_path = os.path.join(pred_saved_dir, file)
        mkdir(pred_saved_path)
        cv2.imwrite(pred_saved_path, pred_im)


def main(args):
    with fluid.dygraph.guard(places):
        test_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])

        if args.model_name == 'UNet':
            model = models.UNet(num_classes=args.num_classes)

        infer(model,
              data_dir=args.data_dir,
              test_list=args.test_list,
              model_dir=args.model_dir,
              transforms=test_transforms)


if __name__ == '__main__':
    args = parse_args()
    env_info = get_environ_info()
    if env_info['place'] == 'cpu':
        places = fluid.CPUPlace()
    else:
        places = fluid.CUDAPlace(0)
    main(args)
