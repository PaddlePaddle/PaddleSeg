# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
File: convert_cityscapes.py
This file is based on https://github.com/mcordts/cityscapesScripts to generate **labelTrainIds.png for training.
Before running, you should download the cityscapes form https://www.cityscapes-dataset.com/ and make the folder
structure as follow:
cityscapes
|
|--leftImg8bit
|  |--train
|  |--val
|  |--test
|
|--gtFine
|  |--train
|  |--val
|  |--test
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
import glob

from cityscapesscripts.preparation.json2labelImg import json2labelImg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate **labelTrainIds.png for training')
    parser.add_argument(
        '--cityscapes_path',
        dest='cityscapes_path',
        help='cityscapes path',
        type=str)

    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='How many processes are used for data conversion',
        type=int,
        default=cpu_count())
    return parser.parse_args()


def gen_labelTrainIds(json_file):
    label_file = json_file.replace("_polygons.json", "_labelTrainIds.png")
    json2labelImg(json_file, label_file, "trainIds")


def main():
    args = parse_args()
    fine_path = os.path.join(args.cityscapes_path, 'gtFine')
    json_files = glob.glob(os.path.join(fine_path, '*', '*', '*_polygons.json'))

    print('generating **_labelTrainIds.png')
    p = Pool(args.num_workers)
    for f in json_files:
        p.apply_async(gen_labelTrainIds, args=(f, ))
    p.close()
    p.join()


if __name__ == '__main__':
    main()
