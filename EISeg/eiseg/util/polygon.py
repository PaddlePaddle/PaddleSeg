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

from enum import Enum

import cv2
import numpy as np
import math
from .regularization import boundary_regularization


class Instructions(Enum):
    No_Instruction = 0
    Polygon_Instruction = 1


def get_polygon(label, sample="Dynamic", img_size=None, building=False):
    results = cv2.findContours(
        image=label, mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_TC89_KCOS)  # 获取内外边界，用RETR_TREE更好表示
    cv2_v = cv2.__version__.split(".")[0]
    contours = results[1] if cv2_v == "3" else results[0]  # 边界
    hierarchys = results[2] if cv2_v == "3" else results[1]  # 隶属信息
    if len(contours) != 0:  # 可能出现没有边界的情况
        polygons = []
        relas = []
        img_shape = label.shape
        for idx, (contour,
                  hierarchy) in enumerate(zip(contours, hierarchys[0])):
            # print(hierarchy)
            # opencv实现边界简化
            epsilon = (0.005 * cv2.arcLength(contour, True)
                       if sample == "Dynamic" else sample)
            if not isinstance(epsilon, float) and not isinstance(epsilon, int):
                epsilon = 0
            # print("epsilon:", epsilon)
            if building is False:
                # -- Douglas-Peucker算法边界简化
                contour = cv2.approxPolyDP(contour, epsilon / 10, True)
            else:
                # -- 建筑边界简化（https://github.com/niecongchong/RS-building-regularization）
                contour = boundary_regularization(contour, img_shape, epsilon)
            # -- 自定义（角度和距离）边界简化
            out = approx_poly_DIY(contour)
            # 给出关系
            rela = (
                idx,  # own
                hierarchy[-1] if hierarchy[-1] != -1 else None, )  # parent
            polygon = []
            for p in out:
                polygon.append(p[0])
            polygons.append(polygon)  # 边界
            relas.append(rela)  # 关系
        for i in range(len(relas)):
            if relas[i][1] != None:  # 有父圈
                for j in range(len(relas)):
                    if relas[j][0] == relas[i][1]:  # i的父圈就是j（i是j的子圈）
                        if polygons[i] is not None and polygons[j] is not None:
                            min_i, min_o = __find_min_point(polygons[i],
                                                            polygons[j])
                            # 改变顺序
                            polygons[i] = __change_list(polygons[i], min_i)
                            polygons[j] = __change_list(polygons[j], min_o)
                            # 连接
                            if min_i != -1 and len(polygons[i]) > 0:
                                polygons[j].extend(polygons[i])  # 连接内圈
                            polygons[i] = None
        polygons = list(filter(None, polygons))  # 清除加到外圈的内圈多边形
        if img_size is not None:
            polygons = check_size_minmax(polygons, img_size)
        return polygons
    else:
        print("没有标签范围，无法生成边界")
        return None


def __change_list(polygons, idx):
    if idx == -1:
        return polygons
    s_p = polygons[:idx]
    polygons = polygons[idx:]
    polygons.extend(s_p)
    polygons.append(polygons[0])  # 闭合圈
    return polygons


def __find_min_point(i_list, o_list):
    min_dis = 1e7
    idx_i = -1
    idx_o = -1
    for i in range(len(i_list)):
        for o in range(len(o_list)):
            dis = math.sqrt((i_list[i][0] - o_list[o][0])**2 + (i_list[i][
                1] - o_list[o][1])**2)
            if dis <= min_dis:
                min_dis = dis
                idx_i = i
                idx_o = o
    return idx_i, idx_o


# 根据三点坐标计算夹角
def __cal_ang(p1, p2, p3):
    eps = 1e-12
    a = math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] -
                                                                         p3[1]))
    b = math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] -
                                                                         p3[1]))
    c = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] -
                                                                         p2[1]))
    ang = math.degrees(math.acos(
        (b**2 - a**2 - c**2) / (-2 * a * c + eps)))  # p2对应
    return ang


# 计算两点距离
def __cal_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# 边界点简化
def approx_poly_DIY(contour, min_dist=10, ang_err=5):
    # print(contour.shape)  # N, 1, 2
    cs = [contour[i][0] for i in range(contour.shape[0])]
    ## 1. 先删除两个相近点与前后两个点角度接近的点
    i = 0
    while i < len(cs):
        try:
            j = (i + 1) if (i != len(cs) - 1) else 0
            if __cal_dist(cs[i], cs[j]) < min_dist:
                last = (i - 1) if (i != 0) else (len(cs) - 1)
                next = (j + 1) if (j != len(cs) - 1) else 0
                ang_i = __cal_ang(cs[last], cs[i], cs[next])
                ang_j = __cal_ang(cs[last], cs[j], cs[next])
                # print(ang_i, ang_j)  # 角度值为-180到+180
                if abs(ang_i - ang_j) < ang_err:
                    # 删除距离两点小的
                    dist_i = __cal_dist(cs[last], cs[i]) + __cal_dist(cs[i],
                                                                      cs[next])
                    dist_j = __cal_dist(cs[last], cs[j]) + __cal_dist(cs[j],
                                                                      cs[next])
                    if dist_j < dist_i:
                        del cs[j]
                    else:
                        del cs[i]
                else:
                    i += 1
            else:
                i += 1
        except:
            i += 1
    ## 2. 再删除夹角接近180度的点
    i = 0
    while i < len(cs):
        try:
            last = (i - 1) if (i != 0) else (len(cs) - 1)
            next = (i + 1) if (i != len(cs) - 1) else 0
            ang_i = __cal_ang(cs[last], cs[i], cs[next])
            if abs(ang_i) > (180 - ang_err):
                del cs[i]
            else:
                i += 1
        except:
            # i += 1
            del cs[i]
    res = np.array(cs).reshape([-1, 1, 2])
    return res


def check_size_minmax(polygons, img_size):
    h_max, w_max = img_size
    for ps in polygons:
        for j in range(len(ps)):
            x, y = ps[j]
            if x < 0:
                x = 0
            elif x > w_max:
                x = w_max
            if y < 0:
                y = 0
            elif y > h_max:
                y = h_max
            ps[j] = np.array([x, y])
    return polygons
