# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
import os.path as osp
import random
import imghdr
import gdal
import numpy as np
from utils import logging
from .base import BaseReader
from .base import get_encoding
from collections import OrderedDict
from PIL import Image


def read_img(img_path):
    img_format = imghdr.what(img_path)
    name, ext = osp.splitext(img_path)
    if img_format == 'tiff' or ext == '.img':
        dataset = gdal.Open(img_path)
        if dataset == None:
            raise Exception('Can not open', img_path)
        im_data = dataset.ReadAsArray()
        return im_data.transpose((1, 2, 0))
    elif img_format == 'png':
        return np.asarray(Image.open(img_path))
    elif ext == '.npy':
        return np.load(img_path)
    else:
        raise Exception('Not support {} image format!'.format(ext))


class Reader(BaseReader):
    """读取数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
        label_list (str): 描述数据集包含的类别信息文件路径。
        transforms (list): 数据集中每个样本的预处理/增强算子。
        num_workers (int): 数据集中样本在预处理过程中的线程或进程数。默认为4。
        buffer_size (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。
        parallel_method (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'
            线程和'process'进程两种方式。默认为'thread'。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list,
                 transforms=None,
                 num_workers=4,
                 buffer_size=100,
                 parallel_method='thread',
                 shuffle=False):
        super(Reader, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = OrderedDict()
        self.labels = list()
        self._epoch = 0

        with open(label_list, encoding=get_encoding(label_list)) as f:
            for line in f:
                item = line.strip()
                self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not osp.exists(full_path_im):
                    raise IOError(
                        'The image file {} is not exist!'.format(full_path_im))
                if not osp.exists(full_path_label):
                    raise IOError('The image file {} is not exist!'.format(
                        full_path_label))
                self.file_list[full_path_im] = full_path_label
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def iterator(self):
        self._epoch += 1
        self._pos = 0
        files = list(self.file_list.keys())
        if self.shuffle:
            random.shuffle(files)
        files = files[:self.num_samples]
        self.num_samples = len(files)
        for f in files:
            label_path = self.file_list[f]
            sample = [f, None, label_path]
            yield sample
