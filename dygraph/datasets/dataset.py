#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid


class Dataset(fluid.io.Dataset):
    def __init__(self,
                 data_dir,
                 num_classes,
                 train_list=None,
                 val_list=None,
                 test_list=None,
                 separator=' ',
                 transforms=None,
                 mode='train'):
        self.data_dir = data_dir
        self.transforms = transforms
        self.file_list = list()
        self.mode = mode
        self.num_classes = num_classes

        if mode.lower() not in ['train', 'eval', 'test']:
            raise Exception(
                "mode should be 'train', 'eval' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("transform is necessary, but it is None.")

        self.data_dir = data_dir
        if mode == 'train':
            if train_list is None:
                raise Exception(
                    'When mode is "train", train_list is need, but it is None.')
            elif not os.path.exists(train_list):
                raise Exception(
                    'train_list is not found: {}'.format(train_list))
            else:
                file_list = train_list
        elif mode == 'eval':
            if val_list is None:
                raise Exception(
                    'When mode is "eval", val_list is need, but it is None.')
            elif not os.path.exists(val_list):
                raise Exception('val_list is not found: {}'.format(val_list))
            else:
                file_list = val_list
        else:
            if test_list is None:
                raise Exception(
                    'When mode is "test", test_list is need, but it is None.')
            elif not os.path.exists(test_list):
                raise Exception('test_list is not found: {}'.format(test_list))
            else:
                file_list = test_list

        with open(file_list, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if mode == 'train' or mode == 'eval':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.data_dir, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.data_dir, items[0])
                    grt_path = os.path.join(self.data_dir, items[1])
                self.file_list.append([image_path, grt_path])

    def __getitem__(self, idx):
        image_path, grt_path = self.file_list[idx]
        im, im_info, label = self.transforms(im=image_path, label=grt_path)
        if self.mode == 'train':
            return im, label
        elif self.mode == 'eval':
            return im, label
        if self.mode == 'test':
            return im, im_info, image_path

    def __len__(self):
        return len(self.file_list)
