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
                                         dtype=np.float)
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
                                         dtype=np.float)

    def _iou(self):
        """
        Intersection over Union (IoU)
        """
        cm_diag = np.diag(self.confusion_matrix)
        iou = cm_diag / (np.sum(self.confusion_matrix, axis=1) + np.sum(
            self.confusion_matrix, axis=0) - cm_diag)
        return iou

    def mean_iou(self):
        """
        Mean Intersection over Union (MIoU)
        """
        iou = self._iou()
        m_iou = np.mean(iou)
        return iou, m_iou

    def frequency_weighted_iou(self):
        """
        Frequency Weighted Intersection over Union (FWIoU)
        """
        frequency = np.sum(
            self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iou = self._iou()
        fw_iou = np.sum(frequency * iou)
        return fw_iou

    def accuracy(self):
        """
        Mean Pixel Accuracy (MPA)
        """
        pa = np.diag(self.confusion_matrix) / np.sum(
            self.confusion_matrix, axis=1)
        mpa = np.mean(pa)
        return pa, mpa

    def kappa(self):
        """
        Kappa coefficient
        """
        cm_sum = np.sum(self.confusion_matrix)
        po = np.sum(np.diag(self.confusion_matrix)) / cm_sum
        pe = np.dot(
            np.sum(self.confusion_matrix, axis=0),
            np.sum(self.confusion_matrix, axis=1)) / (cm_sum**2)
        kappa = (po - pe) / (1 - pe)
        return kappa
