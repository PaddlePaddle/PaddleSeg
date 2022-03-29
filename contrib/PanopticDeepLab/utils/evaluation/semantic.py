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

# ------------------------------------------------------------------------------
# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/evaluation/semantic.py
# Modified by Guowei Chen
# ------------------------------------------------------------------------------

from collections import OrderedDict

import numpy as np


class SemanticEvaluator:
    """
    Evaluate semantic segmentation

    Args:
        num_classes (int): number of classes
        ignore_index (int, optional): value in semantic segmentation ground truth. Predictions for the
        corresponding pixels should be ignored. Default: 255.
    """

    def __init__(self, num_classes, ignore_index=255):
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._N = num_classes + 1  # store ignore label in the last class

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def update(self, pred, gt):
        pred = pred.astype(np.int)
        gt = gt.astype(np.int)
        gt[gt == self._ignore_index] = self._num_classes

        # raw: pred, column: gt
        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N
            **2).reshape(self._N, self._N)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)

        acc_valid = pos_pred > 0
        acc[acc_valid] = tp[acc_valid] / pos_pred[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        return results
