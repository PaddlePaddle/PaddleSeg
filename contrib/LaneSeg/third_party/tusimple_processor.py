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

import paddle.nn as nn
import json
import os
import cv2
import numpy as np

from .lane import LaneEval


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


class TusimpleProcessor:
    def __init__(self,
                 num_classes=2,
                 cut_height=0,
                 thresh=0.6,
                 is_view=False,
                 test_gt_json=None,
                 save_dir='output/'):
        super(TusimpleProcessor, self).__init__()
        self.num_classes = num_classes
        self.cut_height = cut_height
        self.dump_to_json = []
        self.thresh = thresh

        self.save_dir = save_dir
        self.is_view = is_view
        self.test_gt_json = test_gt_json
        self.smooth = True
        self.y_pixel_gap = 10
        self.points_nums = 56
        self.src_height = 720
        self.src_width = 1280
        self.color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 125), (50, 100, 50),
                          (100, 50, 100)]

    def dump_data_to_json(self, output, im_path, run_time):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()

        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            json_pred = {}
            json_pred['lanes'] = []
            json_pred['run_time'] = run_time * 1000
            json_pred['h_sample'] = []
            path_list = im_path[batch].split("/")
            json_pred['raw_file'] = os.path.join(*path_list[-4:])
            for l in lane_coords:
                if len(l) == 0:
                    continue
                json_pred['lanes'].append([])
                for (x, y) in l:
                    json_pred['lanes'][-1].append(int(x))
            for (x, y) in lane_coords[0]:
                json_pred['h_sample'].append(y)
            self.dump_to_json.append(json.dumps(json_pred))

            if self.is_view:
                img = cv2.imread(im_path[batch])
                img_name = '_'.join([x for x in path_list[-4:]])
                saved_path = os.path.join(self.save_dir, 'visual', img_name)
                self.draw(img, lane_coords, saved_path)

    def predict(self, output, im_path):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        lane_coords_list = self.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            img = cv2.imread(im_path)
            img_name = os.path.basename(im_path)
            saved_path = os.path.join(self.save_dir, 'visual_points', img_name)
            self.draw(img, lane_coords, saved_path)

    def bench_one_submit(self):
        output_file = os.path.join(self.save_dir, 'pred.json')
        if output_file is not None:
            mkdir(output_file)
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_rst, acc, fp, fn = LaneEval.bench_one_submit(
            output_file, self.test_gt_json)
        self.dump_to_json = []
        return acc, fp, fn, eval_rst

    def draw(self, img, coords, file_path=None):
        for i, coord in enumerate(coords):
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                cv2.circle(img, (int(x), int(y)), 4,
                           self.color_map[i % self.num_classes], 2)

        if file_path is not None:
            mkdir(file_path)
            cv2.imwrite(file_path, img)

    def get_lane_coords(self, seg_pred):
        lane_coords_list = []
        for batch in range(len(seg_pred)):
            seg = seg_pred[batch]
            lane_coords = self.heatmap2coords(seg)
            for i in range(len(lane_coords)):
                lane_coords[i] = sorted(
                    lane_coords[i], key=lambda pair: pair[1])
            lane_coords_list.append(lane_coords)
        return lane_coords_list

    def process_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [
                i for i, x in reversed(list(enumerate(coordinate))) if x > 0
            ][0]
            lane = coordinate[start:end + 1]
            # The line segment is not continuous
            if any(x < 0 for x in lane):
                gap_start = [
                    i for i, x in enumerate(lane[:-1])
                    if x > 0 and lane[i + 1] < 0
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
                            lane[id] = int((id - gap_start[i]) / gap_width *
                                           lane[gap_end[i]] +
                                           (gap_end[i] - id) / gap_width *
                                           lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def get_coords(self, heat_map):
        dst_height = self.src_height - self.cut_height
        coords = np.zeros(self.points_nums)
        coords[:] = -1
        pointCount = 0
        for i in range(self.points_nums):
            y_coord = dst_height - 10 - i * self.y_pixel_gap
            y = int(y_coord / dst_height * heat_map.shape[0])
            if y < 0:
                break
            prob_line = heat_map[y, :]
            x = np.argmax(prob_line)
            prob = prob_line[x]
            if prob > self.thresh:
                coords[i] = int(x / heat_map.shape[1] * self.src_width)
                pointCount = pointCount + 1
        if pointCount < 2:
            coords = np.zeros(self.points_nums)
        self.process_gap(coords)
        return coords

    def fix_outliers(self, coords):
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
                    coords[ind] = -1
                else:
                    is_outlier = False

    def heatmap2coords(self, seg_pred):
        coordinates = []
        for i in range(self.num_classes - 1):
            heat_map = seg_pred[i + 1]
            if self.smooth:
                heat_map = cv2.blur(
                    heat_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_coords(heat_map)
            start = [i for i, x in enumerate(coords) if x > 0]
            if not start:
                continue
            self.add_coords(coordinates, coords)

        if len(coordinates) == 0:
            coords = np.zeros(self.points_nums)
            self.add_coords(coordinates, coords)
        return coordinates

    def add_coords(self, coordinates, coords):
        sub_lanes = []
        for j in range(self.points_nums):
            y_lane = self.src_height - 10 - j * self.y_pixel_gap
            x_lane = coords[j] if coords[j] > 0 else -2
            sub_lanes.append([x_lane, y_lane])
        coordinates.append(sub_lanes)
