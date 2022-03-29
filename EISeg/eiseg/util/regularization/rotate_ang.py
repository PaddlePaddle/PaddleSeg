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

import math


# 顺时针旋转
def Nrotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)
    dest_x = (src_x - center_x) * math.cos(radian) + \
             (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_y - center_y) * math.cos(radian) - \
             (src_x - center_x) * math.sin(radian) + center_y
    # return (int(dest_x), int(dest_y))
    return (dest_x, dest_y)


# 逆时针旋转
def Srotation_angle_get_coor_coordinates(point, center, angle):
    src_x, src_y = point
    center_x, center_y = center
    radian = math.radians(angle)
    dest_x = (src_x - center_x) * math.cos(radian) - \
             (src_y - center_y) * math.sin(radian) + center_x
    dest_y = (src_x - center_x) * math.sin(radian) + \
             (src_y - center_y) * math.cos(radian) + center_y
    # return [int(dest_x), int(dest_y)]
    return (dest_x, dest_y)
