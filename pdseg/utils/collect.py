#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""A simple attribute dictionary used for representing configuration options."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import codecs
from ast import literal_eval

import yaml
import six


class SegConfig(dict):
    def __init__(self, *args, **kwargs):
        super(SegConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = SegConfig()
        return self[key]

    def __setitem__(self, key, value):
        #
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(SegConfig, self).__setitem__(key, value)

    def update_from_segconfig(self, other):
        if isinstance(other, dict):
            other = SegConfig(other)
        assert isinstance(other, SegConfig)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def check_and_infer(self, reset_dataset=False):
        if self.DATASET.IMAGE_TYPE in ['rgb', 'gray']:
            self.DATASET.DATA_DIM = 3
        elif self.DATASET.IMAGE_TYPE in ['rgba']:
            self.DATASET.DATA_DIM = 4
        else:
            raise KeyError(
                'DATASET.IMAGE_TYPE config error, only support `rgb`, `gray` and `rgba`'
            )
        if self.MEAN is not None:
            self.DATASET.PADDING_VALUE = [x*255.0 for x in self.MEAN]

        if not self.TRAIN_CROP_SIZE:
            raise ValueError(
                'TRAIN_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        if not self.EVAL_CROP_SIZE:
            raise ValueError(
                'EVAL_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        if reset_dataset:
            # Ensure file list is use UTF-8 encoding
            train_sets = codecs.open(self.DATASET.TRAIN_FILE_LIST, 'r',
                                     'utf-8').readlines()
            val_sets = codecs.open(self.DATASET.VAL_FILE_LIST, 'r',
                                   'utf-8').readlines()
            test_sets = codecs.open(self.DATASET.TEST_FILE_LIST, 'r',
                                    'utf-8').readlines()
            self.DATASET.TRAIN_TOTAL_IMAGES = len(train_sets)
            self.DATASET.VAL_TOTAL_IMAGES = len(val_sets)
            self.DATASET.TEST_TOTAL_IMAGES = len(test_sets)

        if self.MODEL.MODEL_NAME == 'icnet' and \
                len(self.MODEL.MULTI_LOSS_WEIGHT) != 3:
            self.MODEL.MULTI_LOSS_WEIGHT = [1.0, 0.4, 0.16]

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        with codecs.open(config_file, 'r', 'utf-8') as file:
            dic = yaml.load(file)
        self.update_from_segconfig(dic)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, SegConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable
