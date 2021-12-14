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

import os.path as osp
import random

from eiseg import pjpath


class ColorMap(object):
    def __init__(self, color_path, shuffle=False):
        self.colors = []
        self.index = 0
        self.usedColors = []
        with open(color_path, "r") as f:
            colors = f.readlines()
        if shuffle:
            random.shuffle(colors)
        self.colors = [[int(x) for x in c.strip().split(",")] for c in colors]

    def get_color(self):
        color = self.colors[self.index]
        self.index = (self.index + 1) % len(self)
        return color

    def __len__(self):
        return len(self.colors)


colorMap = ColorMap(osp.join(pjpath, "config/colormap.txt"))
