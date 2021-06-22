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

import numpy as np


class MSE():
    """
    Only calculate the unknown region if trimap provided.
    """

    def __init__(self):
        self.mse_diffs = 0
        self.count = 0

    def update(self, pred, gt, trimap=None):
        """
        update metric.

        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 255].
            trimap (np.ndarray, optional) The value is in {0, 128, 255}. Default: None.
        """
        if trimap is None:
            trimap = np.ones_like(gt) * 128
        if not (pred.shape == gt.shape == trimap.shape):
            raise ValueError(
                'The shape of `pred`, `gt` and `trimap` should be equal. '
                'but they are {}, {} and {}'.format(pred.shape, gt.shape,
                                                    trimap.shape))
        mask = trimap == 128
        pixels = float(mask.sum())
        gt = gt / 255.
        diff = (pred - gt) * mask
        mse_diff = (diff**2).sum() / pixels if pixels > 0 else 0

        self.mse_diffs += mse_diff
        self.count += 1

    def evaluate(self):
        mse = self.mse_diffs / self.count if self.count > 0 else 0
        return mse


class SAD():
    """
    Only calculate the unknown region if trimap provided.
    """

    def __init__(self):
        self.sad_diffs = 0
        self.count = 0

    def update(self, pred, gt, trimap=None):
        """
        update metric.

        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 255].
            trimap (np.ndarray, optional)L The value is in {0, 128, 255}. Default: None.
        """
        if trimap is None:
            trimap = np.ones_like(gt) * 128
        if not (pred.shape == gt.shape == trimap.shape):
            raise ValueError(
                'The shape of `pred`, `gt` and `trimap` should be equal. '
                'but they are {}, {} and {}'.format(pred.shape, gt.shape,
                                                    trimap.shape))

        mask = trimap == 128
        gt = gt / 255.
        diff = (pred - gt) * mask
        sad_diff = (np.abs(diff)).sum()

        self.sad_diffs += sad_diff
        self.count += 1

    def evaluate(self):
        sad = self.sad_diffs / self.count if self.count > 0 else 0
        return sad
