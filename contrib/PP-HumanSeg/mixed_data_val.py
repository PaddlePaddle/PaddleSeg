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

import numpy as np
import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.core import evaluate
from paddleseg.models import *
from paddleseg.utils import get_sys_env, logger, config_check, utils
from paddleseg.transforms import *
from paddleseg.datasets import Dataset
from datasets.humanseg import HumanSeg


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        '--model_name', dest='model_name', type=str, default=None)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument('--val_roots', dest='val_roots', nargs='+', type=str)
    parser.add_argument(
        '--dataset_weights', dest='dataset_weights', nargs='+', type=float)

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    #========================================= config ======================================
    file_list_name = 'test.txt'
    # file_list_name = 'val.txt'
    val_roots = args.val_roots
    dataset_weights = args.dataset_weights
    class_weights = [0.3, 0.7]

    val_transforms = [
        PaddingByAspectRatio(1.77777778),
        Resize(target_size=[398, 224]),
        Normalize()
    ]

    val_dataset0 = Dataset(
        val_transforms,
        val_roots[0],
        2,
        mode='val',
        val_path=os.path.join(val_roots[0], file_list_name))
    val_dataset1 = Dataset(
        val_transforms,
        val_roots[1],
        2,
        mode='val',
        val_path=os.path.join(val_roots[1], file_list_name))
    val_dataset2 = Dataset(
        val_transforms,
        val_roots[2],
        2,
        mode='val',
        val_path=os.path.join(val_roots[2], file_list_name))
    val_datasets = [val_dataset0, val_dataset1, val_dataset2]
    # val_datasets = [val_dataset0]
    #====================================================================================

    # from paddleseg.models.backbones.lcnet import PPLCNet_x1_0
    # backbone = PPLCNet_x1_0()
    # model_class = manager.MODELS[args.model_name]
    # model = model_class(2, backbone)

    from paddleseg.models.stdcseg import STDCSeg
    from paddleseg.models.backbones import STDC1, STDC2
    backbone = STDC2(
        pretrained='https://bj.bcebos.com/paddleseg/dygraph/STDCNet2.tar.gz')
    model_class = manager.MODELS[args.model_name]
    model = model_class(2, backbone)

    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    logger.info('val_roots {}'.format(str(val_roots)))
    logger.info('dataset_weights {}'.format(str(dataset_weights)))
    logger.info('class_weights {}'.format(str(class_weights)))
    logger.info('file_list_name {}'.format(file_list_name))

    class_ious = []
    for val_dataset in val_datasets:
        mean_iou, acc, class_iou, _, _ = evaluate(
            model, val_dataset, num_workers=args.num_workers)
        class_ious.append(class_iou)

    dataset_ious = []
    for class_iou in class_ious:
        dataset_iou = 0
        for num, class_weight in enumerate(class_weights):
            dataset_iou += class_weight * class_iou[num]
        dataset_ious.append(dataset_iou)
    total_iou = 0
    for num, dataset_iou in enumerate(dataset_ious):
        logger.info("[EVAL] Dataset {} Class IoU: {}\n".format(
            num, str(np.round(class_ious[num], 4))))
        logger.info("[EVAL] Dataset {} IoU: {}\n".format(
            num, str(np.round(dataset_iou, 4))))
        total_iou += dataset_weights[num] * dataset_iou
    logger.info("[EVAL] Total IoU: \n" + str(np.round(total_iou, 4)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
