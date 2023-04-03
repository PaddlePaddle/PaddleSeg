# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import cv2
import numpy as np
import prettytable as pt
import paddle
import paddle.nn.functional as F

from qinspector.utils.logger import setup_logger

logger = setup_logger('CalClassWeights')


def parse_args():
    parser = argparse.ArgumentParser(description='Frequency')
    parser.add_argument(
        '--anno_path',
        help='The path of annotated images',
        required=True,
        type=str)
    parser.add_argument(
        '--temperature',
        help='The temperature of weights',
        default=0.8,
        type=float)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='How many processes are used for data conversion',
        type=int,
        default=cpu_count())
    return parser.parse_args()


def get_class_loss_weights(overall_sample_class_stats, temperature=0.1):
    overall_class_stats = defaultdict(int)
    for _, value in overall_sample_class_stats.items():
        for c, n in value.items():
            c = int(c)
            overall_class_stats[c] += n

    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }

    freq_num = np.asarray(list(overall_class_stats.values()), dtype='float32')
    freq = freq_num / np.sum(freq_num)
    freq = 1 - freq
    freq = F.softmax(paddle.to_tensor(freq / temperature, place='cpu'), axis=-1)

    column_names = ["ClassID", *list(overall_class_stats.keys())]
    table = pt.PrettyTable(column_names)

    table.add_row(["Frequency", * [int(num) for num in freq_num.tolist()]])
    table.add_row(
        ["Weights", * ["{:.2f}".format(num) for num in freq.tolist()]])
    logger.info("Overall Class Frequency Matrix: \n" + str(table))


def get_sample_class(anno_file):
    sample_class_stats = {}
    anno = cv2.imread(anno_file, -1)
    class_list = np.unique(anno)
    sample_class_stats[anno_file] = defaultdict(int)
    for classid in class_list:
        sample_class_stats[anno_file][classid] += (anno == classid).sum()
    return sample_class_stats


def get_file_paths(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def main():
    args = parse_args()
    p = Pool(args.num_workers)

    anno_list = get_file_paths(args.anno_path)
    overall_sample_class_stats = {}
    for anno in anno_list:
        sample_class_stats = p.apply_async(get_sample_class, args=(anno, ))
        overall_sample_class_stats.update(sample_class_stats.get())
    p.close()
    p.join()

    get_class_loss_weights(
        overall_sample_class_stats, temperature=args.temperature)


if __name__ == '__main__':
    main()
