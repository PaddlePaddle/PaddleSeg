# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlepanseg.cvlibs import build_info_dict
from .evaluator import Evaluator


class SemSegEvaluator(Evaluator):
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._N = num_classes + 1  # Store ignore label in the last class

        self.conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def update(self, sample_dict, pp_out_dict):
        pred = pp_out_dict['sem_pred'].squeeze().numpy().astype(np.int64)
        gt = sample_dict['sem_label'].squeeze().numpy().astype(np.int64)
        gt[gt == self.ignore_index] = self.num_classes

        # row: pred, column: gt
        self.conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N
            **2).reshape(self._N, self._N)

    def evaluate(self):
        acc = np.zeros(self.num_classes, dtype=float)
        iou = np.zeros(self.num_classes, dtype=float)
        tp = self.conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self.conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self.conf_matrix[:-1, :-1], axis=1).astype(float)

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

        results = build_info_dict(_type_='metric', sem_metrics=res)
        return results
