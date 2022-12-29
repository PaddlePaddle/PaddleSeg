#!/usr/bin/env python

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os.path as osp
import glob
"""
Cityscapes/
|--gtFine/
|  |--train/
|  |--val/
|  |--test/
|  |--cityscapes_panoptic_train_trainId.json
|  |--cityscapes_panoptic_train_trainId/
|  |--cityscapes_panoptic_val.json
|  |--cityscapes_panoptic_val/
|  |--cityscapes_panoptic_test.json
|  |--cityscapes_panoptic_test/
|--leftImg8bit/
|  |--train/
|  |--val/
|  |--test/
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--sep', type=str, default=' ')
    args = parser.parse_args()

    for subset in ('train', 'val'):
        if subset == 'train':
            subdir = "cityscapes_panoptic_train_trainId"
        else:
            subdir = f"cityscapes_panoptic_{subset}"
        with open(osp.join(args.out_dir, f"{subset}_list.txt"), 'w') as f:
            im_paths = sorted(
                glob.glob(
                    osp.join(args.data_dir, "leftImg8bit", subset, "*",
                             "*.png")))
            gt_paths = sorted(
                glob.glob(osp.join(args.data_dir, "gtFine", subdir, "*.png")))
            if len(im_paths) != len(gt_paths):
                raise RuntimeError
            for im_path, gt_path in zip(im_paths, gt_paths):
                im_rel_path = osp.relpath(im_path, args.data_dir)
                gt_rel_path = osp.relpath(gt_path, args.data_dir)
                f.write(args.sep.join([im_rel_path, gt_rel_path]))
                f.write('\n')
