# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import yaml
import codecs
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="The config for train scheduler")
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str)

    return parser.parse_args()


class train_scheduler:
    def __init__(self, args):
        self.cfg = self._parse_from_yaml(args.config)

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build config'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        return dic

    def run(self, ):
        for c in self.cfg:
            os.system("python train.py  –config {}".format(c))

            if self.cfg["coarse"]:
                pass  # ADD save result for coarse model

    def assemble(self, ):
        """Assemble the training results"""
        pass


if __name__ == "__main__":
    args = parse_args()
    scheduler = train_scheduler(args)
    scheduler.run()
