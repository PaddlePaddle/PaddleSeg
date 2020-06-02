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
import os.path as osp
import math

from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle.fluid as fluid

from datasets import Dataset
import transforms as T
import models
import utils.logging as logging
from utils import get_environ_info
from utils import ConfusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help="Model type for traing, which is one of ('UNet')",
        type=str,
        default='UNet')

    # params of dataset
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        type=int,
        default=2)
    parser.add_argument(
        '--ingore_index',
        dest='ignore_index',
        help=
        'The pixel equaling ignore_index will not be computed during evaluation',
        type=int,
        default=255)

    # params of evaluate
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

    return parser.parse_args()


def evaluate(model,
             eval_dataset=None,
             model_dir=None,
             num_classes=None,
             batch_size=2,
             ignore_index=255,
             epoch_id=None):
    ckpt_path = osp.join(model_dir, 'model')
    para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
    model.set_dict(para_state_dict)
    model.eval()

    data_generator = eval_dataset.generator(
        batch_size=batch_size, drop_last=True)
    total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
    conf_mat = ConfusionMatrix(num_classes, streaming=True)

    logging.info(
        "Start to evaluating(total_samples={}, total_steps={})...".format(
            eval_dataset.num_samples, total_steps))
    for step, data in enumerate(data_generator()):
        images = np.array([d[0] for d in data])
        labels = np.array([d[2] for d in data]).astype('int64')
        images = to_variable(images)
        pred, _ = model(images, labels, mode='eval')

        pred = pred.numpy()
        mask = labels != ignore_index

        conf_mat.calculate(pred=pred, label=labels, ignore=mask)
        _, iou = conf_mat.mean_iou()

        logging.info("[EVAL] Epoch={}, Step={}/{}, iou={}".format(
            epoch_id, step + 1, total_steps, iou))

    category_iou, miou = conf_mat.mean_iou()
    category_acc, macc = conf_mat.accuracy()
    logging.info("[EVAL] #image={} acc={:.4f} IoU={:.4f}".format(
        eval_dataset.num_samples, macc, miou))
    logging.info("[EVAL] Category IoU: " + str(category_iou))
    logging.info("[EVAL] Category Acc: " + str(category_acc))
    logging.info("[EVAL] Kappa:{:.4f} ".format(conf_mat.kappa()))


def main(args):
    eval_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])
    eval_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.val_list,
        transforms=eval_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=False)

    if args.model_name == 'UNet':
        model = models.UNet(num_classes=args.num_classes)

    evaluate(
        model,
        eval_dataset,
        model_dir=args.model_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        ignore_index=args.ignore_index,
    )


if __name__ == '__main__':
    args = parse_args()
    env_info = get_environ_info()
    if env_info['place'] == 'cpu':
        places = fluid.CPUPlace()
    else:
        places = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(places):
        main(args)
