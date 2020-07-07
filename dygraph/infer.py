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
from paddle.fluid.dygraph.parallel import ParallelEnv
import cv2
import tqdm

from datasets import OpticDiscSeg, Cityscapes
import transforms as T
from models import MODELS
import utils
import utils.logging as logging
from utils import get_environ_info


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for testing, which is one of {}'.format(
            str(list(MODELS.keys()))),
        type=str,
        default='UNet')

    # params of dataset
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help=
        "The dataset you want to train, which is one of ('OpticDiscSeg', 'Cityscapes')",
        type=str,
        default='OpticDiscSeg')

    # params of prediction
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[512, 512],
        type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=2)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output/result')

    return parser.parse_args()


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
        im = im[np.newaxis, ...]
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


def main(args):
    env_info = get_environ_info()
    places = fluid.CUDAPlace(ParallelEnv().dev_id) \
        if env_info['place'] == 'cuda' and fluid.is_compiled_with_cuda() \
        else fluid.CPUPlace()

    if args.dataset.lower() == 'opticdiscseg':
        dataset = OpticDiscSeg
    elif args.dataset.lower() == 'cityscapes':
        dataset = Cityscapes
    else:
        raise Exception(
            "The --dataset set wrong. It should be one of ('OpticDiscSeg', 'Cityscapes')"
        )

    with fluid.dygraph.guard(places):
        test_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])
        test_dataset = dataset(transforms=test_transforms, mode='test')

        if args.model_name not in MODELS:
            raise Exception(
                '--model_name is invalid. it should be one of {}'.format(
                    str(list(MODELS.keys()))))
        model = MODELS[args.model_name](num_classes=test_dataset.num_classes)

        infer(
            model,
            model_dir=args.model_dir,
            test_dataset=test_dataset,
            save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
