# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
"""
generate tusimple training dataset
"""
import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir',
        type=str,
        help='The origin path of unzipped tusimple dataset')

    return parser.parse_args()


def process_json_file(json_file_path, src_dir, ori_dst_dir, binary_dst_dir,
                      instance_dst_dir):

    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

    image_nums = len(os.listdir(os.path.join(src_dir, ori_dst_dir)))

    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)

            image_dir = ops.split(info_dict['raw_file'])[0]
            image_dir_split = image_dir.split('/')[1:]
            image_dir_split.append(ops.split(info_dict['raw_file'])[1])
            image_name = '_'.join(image_dir_split)
            image_path = ops.join(src_dir, info_dict['raw_file'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_name_new = '{:s}.png'.format(
                '{:d}'.format(line_index + image_nums).zfill(4))

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros(
                [src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros(
                [src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                cv2.polylines(
                    dst_binary_image,
                    lane_pts,
                    isClosed=False,
                    color=255,
                    thickness=5)
                cv2.polylines(
                    dst_instance_image,
                    lane_pts,
                    isClosed=False,
                    color=lane_index * 50 + 20,
                    thickness=5)

            dst_binary_image_path = ops.join(src_dir, binary_dst_dir,
                                             image_name_new)
            dst_instance_image_path = ops.join(src_dir, instance_dst_dir,
                                               image_name_new)
            dst_rgb_image_path = ops.join(src_dir, ori_dst_dir, image_name_new)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            print('Process {:s} success'.format(image_name))


def gen_sample(src_dir,
               b_gt_image_dir,
               i_gt_image_dir,
               image_dir,
               phase='train',
               split=False):

    label_list = []
    with open('{:s}/{}ing/{}.txt'.format(src_dir, phase, phase), 'w') as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(
                instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                print('image: {:s} corrupt'.format(image_name))
                continue
            else:
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path,
                                               instance_gt_image_path)
                file.write(info + '\n')
                label_list.append(info)
    if phase == 'train' and split:
        np.random.RandomState(0).shuffle(label_list)
        val_list_len = len(label_list) // 10
        val_label_list = label_list[:val_list_len]
        train_label_list = label_list[val_list_len:]
        with open('{:s}/{}ing/train_part.txt'.format(src_dir, phase, phase),
                  'w') as file:
            for info in train_label_list:
                file.write(info + '\n')
        with open('{:s}/{}ing/val_part.txt'.format(src_dir, phase, phase),
                  'w') as file:
            for info in val_label_list:
                file.write(info + '\n')
    return


def process_tusimple_dataset(src_dir):

    traing_folder_path = ops.join(src_dir, 'training')
    testing_folder_path = ops.join(src_dir, 'testing')

    os.makedirs(traing_folder_path, exist_ok=True)
    os.makedirs(testing_folder_path, exist_ok=True)

    for json_label_path in glob.glob('{:s}/label*.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]

        shutil.copyfile(json_label_path,
                        ops.join(traing_folder_path, json_label_name))

    for json_label_path in glob.glob('{:s}/test_label.json'.format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]

        shutil.copyfile(json_label_path,
                        ops.join(testing_folder_path, json_label_name))

    train_gt_image_dir = ops.join('training', 'gt_image')
    train_gt_binary_dir = ops.join('training', 'gt_binary_image')
    train_gt_instance_dir = ops.join('training', 'gt_instance_image')

    test_gt_image_dir = ops.join('testing', 'gt_image')
    test_gt_binary_dir = ops.join('testing', 'gt_binary_image')
    test_gt_instance_dir = ops.join('testing', 'gt_instance_image')

    os.makedirs(os.path.join(src_dir, train_gt_image_dir), exist_ok=True)
    os.makedirs(os.path.join(src_dir, train_gt_binary_dir), exist_ok=True)
    os.makedirs(os.path.join(src_dir, train_gt_instance_dir), exist_ok=True)

    os.makedirs(os.path.join(src_dir, test_gt_image_dir), exist_ok=True)
    os.makedirs(os.path.join(src_dir, test_gt_binary_dir), exist_ok=True)
    os.makedirs(os.path.join(src_dir, test_gt_instance_dir), exist_ok=True)

    for json_label_path in glob.glob('{:s}/*.json'.format(traing_folder_path)):
        process_json_file(json_label_path, src_dir, train_gt_image_dir,
                          train_gt_binary_dir, train_gt_instance_dir)

    gen_sample(src_dir, train_gt_binary_dir, train_gt_instance_dir,
               train_gt_image_dir, 'train', True)


if __name__ == '__main__':
    args = init_args()

    process_tusimple_dataset(args.src_dir)
