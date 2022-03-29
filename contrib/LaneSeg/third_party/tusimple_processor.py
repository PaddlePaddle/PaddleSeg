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

import cv2
import json
import paddle.nn as nn

from .lane import LaneEval
from .get_lane_coords import LaneProcessor


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


class TusimpleProcessor:
    def __init__(self,
                 num_classes=2,
                 ori_shape=(720, 1280),
                 cut_height=0,
                 thresh=0.6,
                 test_gt_json=None,
                 save_dir='output/'):
        super(TusimpleProcessor, self).__init__()
        self.num_classes = num_classes
        self.dump_to_json = []
        self.save_dir = save_dir
        self.test_gt_json = test_gt_json
        self.color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                          (255, 0, 255), (0, 255, 125), (50, 100, 50),
                          (100, 50, 100)]
        self.laneProcessor = LaneProcessor(
            num_classes=self.num_classes,
            ori_shape=ori_shape,
            cut_height=cut_height,
            y_pixel_gap=10,
            points_nums=56,
            thresh=thresh,
            smooth=True)

    def dump_data_to_json(self,
                          output,
                          im_path,
                          run_time=0,
                          is_dump_json=True,
                          is_view=False):
        seg_pred = output[0]
        seg_pred = nn.functional.softmax(seg_pred, axis=1)
        seg_pred = seg_pred.numpy()
        lane_coords_list = self.laneProcessor.get_lane_coords(seg_pred)

        for batch in range(len(seg_pred)):
            lane_coords = lane_coords_list[batch]
            path_list = im_path[batch].split("/")
            if is_dump_json:
                json_pred = {}
                json_pred['lanes'] = []
                json_pred['run_time'] = run_time * 1000
                json_pred['h_sample'] = []

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

            if is_view:
                img = cv2.imread(im_path[batch])
                if is_dump_json:
                    img_name = '_'.join([x for x in path_list[-4:]])
                    sub_dir = 'visual_eval'
                else:
                    img_name = os.path.basename(im_path[batch])
                    sub_dir = 'visual_points'
                saved_path = os.path.join(self.save_dir, sub_dir, img_name)
                self.draw(img, lane_coords, saved_path)

    def predict(self, output, im_path):
        self.dump_data_to_json(
            output, [im_path], is_dump_json=False, is_view=True)

    def bench_one_submit(self, local_rank):
        output_file = os.path.join(self.save_dir, str(local_rank), 'pred.json')
        if output_file is not None:
            mkdir(output_file)
        with open(output_file, "w+") as f:
            for line in self.dump_to_json:
                print(line, end="\n", file=f)

        eval_rst, acc, fp, fn = LaneEval.bench_one_submit(output_file,
                                                          self.test_gt_json)
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
