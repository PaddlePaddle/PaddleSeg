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

import pickle
import sys
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize data distribution before training.')
    parser.add_argument(
        '--pkl_path',
        dest='pkl_path',
        help='Path of img_pixel_statistics.pkl',
        default=None,
        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.pkl_path
    with open(path, 'rb') as f:
        percentage, img_value_num = pickle.load(f)

    for k in range(len(img_value_num)):
        print('channel = {}'.format(k))
        plt.bar(
            list(range(len(img_value_num[k]))),
            img_value_num[k],
            width=1,
            log=True)
        plt.xlabel('image value')
        plt.ylabel('number')
        plt.title('channel={}'.format(k))
        plt.show()
