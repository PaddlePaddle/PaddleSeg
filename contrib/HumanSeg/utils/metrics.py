# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix


class ConfusionMatrix(object):
    """
        Confusion Matrix for segmentation evaluation
    """

    def __init__(self, num_classes=2, streaming=False):
        self.confusion_matrix = np.zeros([num_classes, num_classes],
                                         dtype='int64')
        self.num_classes = num_classes
        self.streaming = streaming

    def calculate(self, pred, label, ignore=None):
        # If not in streaming mode, clear matrix everytime when call `calculate`
        if not self.streaming:
            self.zero_matrix()

        label = np.transpose(label, (0, 2, 3, 1))
        ignore = np.transpose(ignore, (0, 2, 3, 1))
        mask = np.array(ignore) == 1

        label = np.asarray(label)[mask]
        pred = np.asarray(pred)[mask]
        one = np.ones_like(pred)
        # Accumuate ([row=label, col=pred], 1) into sparse matrix
        spm = csr_matrix((one, (label, pred)),
                         shape=(self.num_classes, self.num_classes))
        spm = spm.todense()
        self.confusion_matrix += spm

    def zero_matrix(self):
        """ Clear confusion matrix """
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes],
                                         dtype='int64')

    def mean_iou(self):
        iou_list = []
        avg_iou = 0
        # TODO: use numpy sum axis api to simpliy
        vji = np.zeros(self.num_classes, dtype=int)
        vij = np.zeros(self.num_classes, dtype=int)
        for j in range(self.num_classes):
            v_j = 0
            for i in range(self.num_classes):
                v_j += self.confusion_matrix[j][i]
            vji[j] = v_j

        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        for c in range(self.num_classes):
            total = vji[c] + vij[c] - self.confusion_matrix[c][c]
            if total == 0:
                iou = 0
            else:
                iou = float(self.confusion_matrix[c][c]) / total
            avg_iou += iou
            iou_list.append(iou)
        avg_iou = float(avg_iou) / float(self.num_classes)
        return np.array(iou_list), avg_iou

    def accuracy(self):
        total = self.confusion_matrix.sum()
        total_right = 0
        for c in range(self.num_classes):
            total_right += self.confusion_matrix[c][c]
        if total == 0:
            avg_acc = 0
        else:
            avg_acc = float(total_right) / total

        vij = np.zeros(self.num_classes, dtype=int)
        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        acc_list = []
        for c in range(self.num_classes):
            if vij[c] == 0:
                acc = 0
            else:
                acc = self.confusion_matrix[c][c] / float(vij[c])
            acc_list.append(acc)
        return np.array(acc_list), avg_acc

    def kappa(self):
        vji = np.zeros(self.num_classes)
        vij = np.zeros(self.num_classes)
        for j in range(self.num_classes):
            v_j = 0
            for i in range(self.num_classes):
                v_j += self.confusion_matrix[j][i]
            vji[j] = v_j

        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        total = self.confusion_matrix.sum()

        # avoid spillovers
        # TODO: is it reasonable to hard code 10000.0?
        total = float(total) / 10000.0
        vji = vji / 10000.0
        vij = vij / 10000.0

        tp = 0
        tc = 0
        for c in range(self.num_classes):
            tp += vji[c] * vij[c]
            tc += self.confusion_matrix[c][c]

        tc = tc / 10000.0
        pe = tp / (total * total)
        po = tc / total

        kappa = (po - pe) / (1 - pe)
        return kappa
