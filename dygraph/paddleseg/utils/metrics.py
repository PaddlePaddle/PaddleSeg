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

import numpy as np
from scipy.sparse import csr_matrix


class ConfusionMatrix(object):
    """
        Confusion Matrix for segmentation evaluation.

        The calculation method of metrics (mIoU, Accuracy, Kappa) refers to
        Hao S, Zhou Y, Guo Y. A Brief Survey on Semantic Segmentation with
        Deep Learning[J]. Neurocomputing, 2020.
        (https://www.sciencedirect.com/science/article/abs/pii/S0925231220305476)
        or
        Garcia-Garcia A, Orts-Escolano S, Oprea S, et al. A review on
        deep learning techniques applied to semantic segmentation[J].
        arXiv preprint arXiv:1704.06857, 2017.
        (https://arxiv.org/pdf/1704.06857.pdf)
    """

    def __init__(self, num_classes=2, streaming=False):
        self.confusion_matrix = np.zeros([num_classes, num_classes],
                                         dtype='int64')
        self.num_classes = num_classes
        self.streaming = streaming

    def calculate(self, pred, label, mask):
        """
        Calculate confusion matrix.

        Args:
            pred (np.ndarray): The prediction of input image by model.
            label (np.ndarray): The ground truth of input image.
            mask (np.ndarray): The mask which pixel is valid. The dtype should be bool.
        """
        # If not in streaming mode, clear matrix everytime when call `calculate`
        if not self.streaming:
            self.zero_matrix()

        pred = np.squeeze(pred)
        label = np.squeeze(label)
        mask = np.squeeze(mask)

        if not pred.shape == label.shape == mask.shape:
            raise ValueError(
                'Shape of `pred`, `label` and `mask` should be equal, '
                'but there are {}, {} and {}.'.format(pred.shape, label.shape,
                                                      mask.shape))

        label = label[mask]
        pred = pred[mask]
        one = np.ones_like(pred).astype('int64')
        # Accumuate ([row=label, col=pred], 1) into sparse
        spm = csr_matrix((one, (label, pred)),
                         shape=(self.num_classes, self.num_classes))
        spm = spm.todense()
        self.confusion_matrix += spm

    def zero_matrix(self):
        """ Clear confusion matrix """
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes],
                                         dtype='int64')

    def mean_iou(self):
        """
        Calculate mIoU and Category IoU.

        Mean Intersection over Union (mIoU): This is the standard metric for
        segmentation purposes. It computes a ratio between the intersection
        and the union of two sets, in our case the ground truth and our
        predicted segmentation.

        Category IoU: Computing mIoU for each class.

        Returns:
            list(numpy.ndarray, float), a list of Category IoU and mIoU.
        """
        iou_list = []
        miou = 0
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
            miou += iou
            iou_list.append(iou)
        miou = float(miou) / float(self.num_classes)
        return np.array(iou_list), miou

    def accuracy(self):
        """
        Calculate Category Accuracy and Pixel Accuracy.

        Pixel Accuracy (PA): Computing a ratio between the amount of
        properly classified pixels and the total number of them.

        Category Accuracy: Computing Pixel Accuracy for each class.

        Returns:
            list(numpy.ndarray, float), a list of Category Accuracy and Pixel Accuracy.
        """
        total = self.confusion_matrix.sum()
        total_right = 0
        for c in range(self.num_classes):
            total_right += self.confusion_matrix[c][c]
        if total == 0:
            pixel_acc = 0
        else:
            pixel_acc = float(total_right) / total

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
        return np.array(acc_list), pixel_acc

    def kappa(self):
        """
        Calculate Cohen's kappa coefficient.

        Cohen's kappa coefficient: A statistic that is used to measure
        inter-rater reliability (and also Intra-rater reliability) for
        qualitative (categorical) items. Refer to
        https://en.wikipedia.org/wiki/Cohen%27s_kappa

        Returns
            (Float). Kappa coefficient.
        """
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
