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
"""
This code is based on https://github.com/niecongchong/RS-building-regularization
Ths copyright of niecongchong/RS-building-regularization is as follows:
Apache License [see LICENSE for details]
"""

import numpy as np
import math


# 计算两点距离
def cal_dist(point_1, point_2):
    dist = np.sqrt(np.sum(np.power((point_1 - point_2), 2)))
    return dist


# 计算两条线的夹角
def cal_ang(point_1, point_2, point_3):
    def _cal_pp(p_1, p_2):
        return math.sqrt((p_1[0] - p_2[0]) * (p_1[0] - p_2[0]) + (p_1[1] - p_2[
            1]) * (p_1[1] - p_2[1]))

    a = _cal_pp(point_2, point_3)
    b = _cal_pp(point_1, point_3)
    c = _cal_pp(point_1, point_2)
    B = math.degrees(math.acos((b**2 - a**2 - c**2) / (-2 * a * c)))
    return B


# 计算线条的方位角
def cal_azimuth(point_0, point_1):
    x1, y1 = point_0
    x2, y2 = point_1
    if x1 < x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x2 - x1))
            ang = ang * 180 / math.pi
            return ang
        elif y1 > y2:
            ang = math.atan((y1 - y2) / (x2 - x1))
            ang = ang * 180 / math.pi
            return 90 + (90 - ang)
        elif y1 == y2:
            return 0
    elif x1 > x2:
        if y1 < y2:
            ang = math.atan((y2 - y1) / (x1 - x2))
            ang = ang * 180 / math.pi
            return 90 + (90 - ang)
        elif y1 > y2:
            ang = math.atan((y1 - y2) / (x1 - x2))
            ang = ang * 180 / math.pi
            return ang
        elif y1 == y2:
            return 0
    elif x1 == x2:
        return 90
