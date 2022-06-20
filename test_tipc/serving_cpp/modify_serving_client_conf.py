# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import argparse

# For feed_var in config file, change the feed_type to 20 and shape to 1.


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--conf_path",
        help="The path of conf file.",
        type=str,
        default="./serving_client/serving_client_conf.prototxt")
    return parser.parse_args()


def modify_serving_client_conf(conf_path):
    lines = open(conf_path).readlines()
    feat_type_str = "  feed_type: "
    shape_str = "  shape: "
    with open(conf_path, 'w') as of:
        for line in lines:
            if line.startswith(feat_type_str):
                line = feat_type_str + "20\n"
            elif line.startswith(shape_str):
                line = shape_str + "1\n"
            of.write(line)


if __name__ == '__main__':
    args = parse_args()
    modify_serving_client_conf(args.conf_path)
