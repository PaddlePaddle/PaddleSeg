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

import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.core import evaluate
from paddleseg.models import *
from paddleseg.utils import get_sys_env, logger, config_check, utils
from paddleseg.transforms import *
from datasets.humanseg import HumanSeg
from datasets.multi_dataset import MultiDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
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

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    dataset_root_list = [
        "data/portrait2600",
        "data/matting_human_half",
        "/ssd3/chulutao/humanseg",
    ]
    valset_weight = [0.2, 0.2, 0.6]
    valset_class_weight = [0.3, 0.7]
    data_ratio = [10, 1, 1]

    val_transforms = [
        PaddingByAspectRatio(1.77777778),
        Resize(target_size=[398, 224]),
        Normalize()
    ]

    val_dataset0 = MultiDataset(
        val_transforms,
        dataset_root_list,
        valset_weight,
        valset_class_weight,
        data_ratio=data_ratio,
        mode='val',
        num_classes=2,
        val_test_set_rank=0)
    val_dataset1 = MultiDataset(
        val_transforms,
        dataset_root_list,
        valset_weight,
        valset_class_weight,
        data_ratio=data_ratio,
        mode='val',
        num_classes=2,
        val_test_set_rank=1)
    val_dataset2 = MultiDataset(
        val_transforms,
        dataset_root_list,
        valset_weight,
        valset_class_weight,
        data_ratio=data_ratio,
        mode='val',
        num_classes=2,
        val_test_set_rank=2)

    model = ShuffleNetV2(align_corners=False, num_classes=2)

    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    logger.info('dataset_root_list {}'.format(str(dataset_root_list)))
    logger.info('valset_weight {}'.format(str(valset_weight)))
    logger.info('valset_class_weight {}'.format(str(valset_class_weight)))
    logger.info('data_ratio {}'.format(str(data_ratio)))

    mean_iou, acc, class_iou0, _, _ = evaluate(
        model, val_dataset0, num_workers=args.num_workers)
    mean_iou, acc, class_iou1, _, _ = evaluate(
        model, val_dataset1, num_workers=args.num_workers)
    mean_iou, acc, class_iou2, _, _ = evaluate(
        model, val_dataset2, num_workers=args.num_workers)

    dataset0_iou = valset_class_weight[0] * class_iou0[0] + valset_class_weight[
        1] * class_iou0[1]
    dataset1_iou = valset_class_weight[0] * class_iou1[0] + valset_class_weight[
        1] * class_iou1[1]
    dataset2_iou = valset_class_weight[0] * class_iou2[0] + valset_class_weight[
        1] * class_iou2[1]
    total_iou = valset_weight[0] * dataset0_iou + valset_weight[
        1] * dataset1_iou + valset_weight[2] * dataset2_iou
    logger.info("[EVAL] Dataset 0 Class IoU: \n" + str(np.round(class_iou0, 4)))
    logger.info("[EVAL] Dataset 1 Class IoU: \n" + str(np.round(class_iou1, 4)))
    logger.info("[EVAL] Dataset 2 Class IoU: \n" + str(np.round(class_iou2, 4)))
    logger.info("[EVAL] Total IoU: \n" + str(np.round(total_iou, 4)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
