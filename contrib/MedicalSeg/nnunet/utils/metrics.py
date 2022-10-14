# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from medpy import metric


class ConfusionMatrix:
    def __init__(self, test=None, reference=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):
        self.test = test
        self.reset()

    def set_reference(self, reference):
        self.reference = reference
        self.reset()

    def reset(self):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        if self.test is None or self.reference is None:
            raise ValueError(
                "'test' and 'reference' must both be set to compute confusion matrix."
            )

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):
        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):
        for case in (self.test_empty, self.test_full, self.reference_empty,
                     self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


def dice(test=None,
         reference=None,
         confusion_matrix=None,
         nan_for_nonexisting=True,
         **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)
    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None,
            reference=None,
            confusion_matrix=None,
            nan_for_nonexisting=True,
            **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(tp / (tp + fp + fn))


def precision(test=None,
              reference=None,
              confusion_matrix=None,
              nan_for_nonexisting=True,
              **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)
    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(tp / (tp + fp))


def sensitivity(test=None,
                reference=None,
                confusion_matrix=None,
                nan_for_nonexisting=True,
                **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(tp / (tp + fn))


def recall(test=None,
           reference=None,
           confusion_matrix=None,
           nan_for_nonexisting=True,
           **kwargs):
    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting,
                       **kwargs)


def specificity(test=None,
                reference=None,
                confusion_matrix=None,
                nan_for_nonexisting=True,
                **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None,
           reference=None,
           confusion_matrix=None,
           nan_for_nonexisting=True,
           beta=1.,
           **kwargs):
    precision_ = precision(test, reference, confusion_matrix,
                           nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)
    return (1 + beta * beta) * precision_ * recall_ / (
        (beta * beta * precision_) + recall_)


def false_positive_rate(test=None,
                        reference=None,
                        confusion_matrix=None,
                        nan_for_nonexisting=True,
                        **kwargs):
    return 1 - specificity(test, reference, confusion_matrix,
                           nan_for_nonexisting)


def false_omission_rate(test=None,
                        reference=None,
                        confusion_matrix=None,
                        nan_for_nonexisting=True,
                        **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.
    return float(fn / (fn + tn))


def false_negative_rate(test=None,
                        reference=None,
                        confusion_matrix=None,
                        nan_for_nonexisting=True,
                        **kwargs):
    return 1 - sensitivity(test, reference, confusion_matrix,
                           nan_for_nonexisting)


def true_negative_rate(test=None,
                       reference=None,
                       confusion_matrix=None,
                       nan_for_nonexisting=True,
                       **kwargs):
    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None,
                         reference=None,
                         confusion_matrix=None,
                         nan_for_nonexisting=True,
                         **kwargs):
    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None,
                              reference=None,
                              confusion_matrix=None,
                              nan_for_nonexisting=True,
                              **kwargs):
    return 1 - false_omission_rate(test, reference, confusion_matrix,
                                   nan_for_nonexisting)


def total_positives_test(test=None,
                         reference=None,
                         confusion_matrix=None,
                         **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return tp + fp


def total_negatives_test(test=None,
                         reference=None,
                         confusion_matrix=None,
                         **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return tn + fn


def total_positives_reference(test=None,
                              reference=None,
                              confusion_matrix=None,
                              **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return tp + fn


def total_negatives_reference(test=None,
                              reference=None,
                              confusion_matrix=None,
                              **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    return tn + fp


def hausdorff_distance(test=None,
                       reference=None,
                       confusion_matrix=None,
                       nan_for_nonexisting=True,
                       voxel_spacing=None,
                       connectivity=1,
                       **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0
    test, reference = confusion_matrix.test, confusion_matrix.reference
    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None,
                          reference=None,
                          confusion_matrix=None,
                          nan_for_nonexisting=True,
                          voxel_spacing=None,
                          connectivity=1,
                          **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0
    test, reference = confusion_matrix.test, confusion_matrix.reference
    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None,
                         reference=None,
                         confusion_matrix=None,
                         nan_for_nonexisting=True,
                         voxel_spacing=None,
                         connectivity=1,
                         **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0
    test, reference = confusion_matrix.test, confusion_matrix.reference
    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None,
                                   reference=None,
                                   confusion_matrix=None,
                                   nan_for_nonexisting=True,
                                   voxel_spacing=None,
                                   connectivity=1,
                                   **kwargs):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence(
    )
    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0
    test, reference = confusion_matrix.test, confusion_matrix.reference
    return metric.assd(test, reference, voxel_spacing, connectivity)


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}
