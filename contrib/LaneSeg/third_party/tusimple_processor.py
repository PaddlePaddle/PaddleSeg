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

import os

import json
import cv2
import paddle.nn as nn

from .lane import LaneEval
from .get_lane_coords import get_lane_coords


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
        self.ori_shape = (720, 1280)
        self.color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 125), (50, 100, 50),
                          (100, 50, 100)]

    def dump_data_to_json(self, output, im_path, run_time):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        lane_coords_list = get_lane_coords(
            seg_pred, self.ori_shape, self.cut_height, self.y_pixel_gap,
            self.num_classes, self.smooth, self.points_nums, self.thresh)

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
        lane_coords_list = get_lane_coords(
            seg_pred, self.ori_shape, self.cut_height, self.y_pixel_gap,
            self.num_classes, self.smooth, self.points_nums, self.thresh)

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
