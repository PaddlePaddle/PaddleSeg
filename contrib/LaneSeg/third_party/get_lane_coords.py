#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# this code is based on
# https://github.com/ZJULearning/resa/blob/main/datasets/tusimple.py

import cv2
import numpy as np


def get_lane_coords(seg_pred, ori_shape, cut_height, y_pixel_gap, num_classes,
                    smooth, points_nums, thresh):
    lane_coords_list = []
    for batch in range(len(seg_pred)):
        seg = seg_pred[batch]
        lane_coords = heatmap2coords(seg, ori_shape, cut_height, y_pixel_gap,
                                     num_classes, smooth, points_nums, thresh)
        for i in range(len(lane_coords)):
            lane_coords[i] = sorted(lane_coords[i], key=lambda pair: pair[1])
        lane_coords_list.append(lane_coords)
    return lane_coords_list


def process_gap(coordinate):
    if any(x > 0 for x in coordinate):
        start = [i for i, x in enumerate(coordinate) if x > 0][0]
        end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
        lane = coordinate[start:end + 1]
        # The line segment is not continuous
        if any(x < 0 for x in lane):
            gap_start = [
                i for i, x in enumerate(lane[:-1]) if x > 0 and lane[i + 1] < 0
            ]
            gap_end = [
                i + 1 for i, x in enumerate(lane[:-1])
                if x < 0 and lane[i + 1] > 0
            ]
            gap_id = [i for i, x in enumerate(lane) if x < 0]
            if len(gap_start) == 0 or len(gap_end) == 0:
                return coordinate
            for id in gap_id:
                for i in range(len(gap_start)):
                    if i >= len(gap_end):
                        return coordinate
                    if id > gap_start[i] and id < gap_end[i]:
                        gap_width = float(gap_end[i] - gap_start[i])
                        # line interpolation
                        lane[id] = int(
                            (id - gap_start[i]) / gap_width * lane[gap_end[i]] +
                            (gap_end[i] - id) / gap_width * lane[gap_start[i]])
            if not all(x > 0 for x in lane):
                print("Gaps still exist!")
            coordinate[start:end + 1] = lane
    return coordinate


def get_coords(heat_map, ori_shape, cut_height, points_nums, y_pixel_gap,
               thresh):
    dst_height = ori_shape[1] - cut_height
    coords = np.zeros(points_nums)
    coords[:] = -2
    pointCount = 0
    for i in range(points_nums):
        y_coord = dst_height - 10 - i * y_pixel_gap
        y = int(y_coord / dst_height * heat_map.shape[0])
        if y < 0:
            break
        prob_line = heat_map[y, :]
        x = np.argmax(prob_line)
        prob = prob_line[x]
        if prob > thresh:
            coords[i] = int(x / heat_map.shape[1] * ori_shape[0])
            pointCount = pointCount + 1
    if pointCount < 2:
        coords[:] = -2
    process_gap(coords)
    return coords


def fix_outliers(coords):
    data = [x for i, x in enumerate(coords) if x > 0]
    index = [i for i, x in enumerate(coords) if x > 0]
    if len(data) == 0:
        return coords
    diff = []
    is_outlier = False
    n = 1
    x_gap = abs((data[-1] - data[0]) / (1.0 * (len(data) - 1)))
    for idx, dt in enumerate(data):
        if is_outlier == False:
            t = idx - 1
            n = 1
        if idx == 0:
            diff.append(0)
        else:
            diff.append(abs(data[idx] - data[t]))
            if abs(data[idx] - data[t]) > n * (x_gap * 1.5):
                n = n + 1
                is_outlier = True
                ind = index[idx]
                coords[ind] = -2
            else:
                is_outlier = False


def heatmap2coords(seg_pred, ori_shape, cut_height, y_pixel_gap, num_classes,
                   smooth, points_nums, thresh):
    coordinates = []
    for i in range(num_classes - 1):
        heat_map = seg_pred[i + 1]
        if smooth:
            heat_map = cv2.blur(
                heat_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
        coords = get_coords(heat_map, ori_shape, cut_height, points_nums,
                            y_pixel_gap, thresh)
        indexes = [i for i, x in enumerate(coords) if x > 0]
        if not indexes:
            continue
        add_coords(coordinates, coords, points_nums, ori_shape[1], y_pixel_gap)

    if len(coordinates) == 0:
        coords = np.zeros(points_nums)
        add_coords(coordinates, coords, points_nums, ori_shape[1], y_pixel_gap)
    return coordinates


def add_coords(coordinates, coords, points_nums, src_height, y_pixel_gap):
    sub_lanes = []
    for j in range(points_nums):
        y_lane = src_height - 10 - j * y_pixel_gap
        x_lane = coords[j] if coords[j] > 0 else -2
        sub_lanes.append([x_lane, y_lane])
    coordinates.append(sub_lanes)
