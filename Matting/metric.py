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

# Grad and Conn is refer to https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/evaluate.py
# Output of `Grad` is sightly different from the MATLAB version provided by Adobe (less than 0.1%)
# Output of `Conn` is smaller than the MATLAB version (~5%, maybe MATLAB has a different algorithm)
# So do not report results calculated by these functions in your paper.
# Evaluate your inference with the MATLAB file `DIM_evaluation_code/evaluate.m`.

import numpy as np
import scipy.ndimage
import numpy as np
from skimage.measure import label


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
            pred (np.ndarray): The value range is [0., 255.].
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
        pred = pred / 255.
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
            pred (np.ndarray): The value range is [0., 255.].
            gt (np.ndarray): The value range is [0., 255.].
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
        pred = pred / 255.
        gt = gt / 255.
        diff = (pred - gt) * mask
        sad_diff = (np.abs(diff)).sum()
        sad_diff /= 1000

        self.sad_diffs += sad_diff
        self.count += 1

    def evaluate(self):
        sad = self.sad_diffs / self.count if self.count > 0 else 0
        return sad


class Grad():
    """
    Only calculate the unknown region if trimap provided.

    Refer to: https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/evaluate.py#L46
    """

    def __init__(self):
        self.grad_diffs = 0
        self.count = 0

    def gauss(self, x, sigma):
        y = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        return y

    def dgauss(self, x, sigma):
        y = -x * self.gauss(x, sigma) / (sigma**2)
        return y

    def gaussgradient(self, im, sigma):
        epsilon = 1e-2
        halfsize = np.ceil(sigma * np.sqrt(
            -2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
        size = 2 * halfsize + 1
        hx = np.zeros((size, size))
        for i in range(0, size):
            for j in range(0, size):
                u = [i - halfsize, j - halfsize]
                hx[i, j] = self.gauss(u[0], sigma) * self.dgauss(u[1], sigma)

        hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
        hy = hx.transpose()

        gx = scipy.ndimage.convolve(im, hx, mode='nearest')
        gy = scipy.ndimage.convolve(im, hy, mode='nearest')

        return gx, gy

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

        pred_x, pred_y = self.gaussgradient(pred, 1.4)
        gt_x, gt_y = self.gaussgradient(gt, 1.4)

        pred_amp = np.sqrt(pred_x**2 + pred_y**2)
        gt_amp = np.sqrt(gt_x**2 + gt_y**2)

        error_map = (pred_amp - gt_amp)**2
        diff = np.sum(error_map[mask])

        self.grad_diffs += diff / 1000.
        self.count += 1

    def evaluate(self):
        grad = self.grad_diffs / self.count if self.count > 0 else 0
        return grad


class Conn():
    """
    Only calculate the unknown region if trimap provided.

    Refer to: https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/evaluate.py#L69
    """

    def __init__(self):
        self.conn_diffs = 0
        self.count = 0

    def getLargestCC(self, segmentation):
        labels = label(segmentation, connectivity=1)
        largestCC = labels == np.argmax(np.bincount(labels.flat))
        return largestCC

    def update(self, pred, gt, trimap=None, step=0.1):
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
        h, w = pred.shape

        thresh_steps = list(np.arange(0, 1 + step, step))
        l_map = np.ones_like(pred, dtype=np.float) * -1
        for i in range(1, len(thresh_steps)):
            pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
            gt_alpha_thresh = (gt >= thresh_steps[i]).astype(np.int)

            omega = self.getLargestCC(
                pred_alpha_thresh * gt_alpha_thresh).astype(np.int)
            flag = ((l_map == -1) & (omega == 0)).astype(np.int)
            l_map[flag == 1] = thresh_steps[i - 1]

        l_map[l_map == -1] = 1

        pred_d = pred - l_map
        gt_d = gt - l_map
        pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
        gt_phi = 1 - gt_d * (gt_d >= 0.15).astype(np.int)
        diff = np.sum(np.abs(pred_phi - gt_phi)[mask])

        self.conn_diffs += diff / 1000.
        self.count += 1

    def evaluate(self):
        conn = self.conn_diffs / self.count if self.count > 0 else 0
        return conn
