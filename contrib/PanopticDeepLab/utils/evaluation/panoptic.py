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
# Reference: https://github.com/mcordts/cityscapesScripts/blob/aeb7b82531f86185ce287705be28f452ba3ddbb8/cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py
# Modified by Guowei Chen
# ------------------------------------------------------------------------------

from collections import defaultdict, OrderedDict

import numpy as np

OFFSET = 256 * 256 * 256


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pd_stat_cat):
        self.iou += pd_stat_cat.iou
        self.tp += pd_stat_cat.tp
        self.fp += pd_stat_cat.fp
        self.fn += pd_stat_cat.fn
        return self

    def __repr__(self):
        s = 'iou: ' + str(self.iou) + ' tp: ' + str(self.tp) + ' fp: ' + str(
            self.fp) + ' fn: ' + str(self.fn)
        return s


class PQStat():
    def __init__(self, num_classes):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.num_classes = num_classes

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pd_stat):
        for label, pq_stat_cat in pd_stat.pq_per_cat.items():
            self.pd_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, isthing=None, thing_list=None):
        """
        Calculate the average pq for all and every class.

        Args:
            num_classes (int): number of classes.
            isthing (bool|None, optional): calculate average pq for thing class if isthing is True,
                for stuff class if isthing is False and for all if isthing is None. Default: None. Default: None.
            thing_list (list|None, optional): A list of thing class. It should be provided when isthing is equal to True or False. Default: None.
        """
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label in range(self.num_classes):
            if isthing is not None:
                if isthing:
                    if label not in thing_list:
                        continue
                else:
                    if label in thing_list:
                        continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)

            per_class_results[label] = {
                'pq': pq_class,
                'sq': sq_class,
                'rq': rq_class
            }
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {
            'pq': pq / n,
            'sq': sq / n,
            'rq': rq / n,
            'n': n
        }, per_class_results


class PanopticEvaluator:
    """
    Evaluate semantic segmentation
    """

    def __init__(self,
                 num_classes,
                 thing_list,
                 ignore_index=255,
                 label_divisor=1000):
        self.pq_stat = PQStat(num_classes)
        self.num_classes = num_classes
        self.thing_list = thing_list
        self.ignore_index = ignore_index
        self.label_divisor = label_divisor

    def update(self, pred, gt):
        # get the labels and counts for the pred and gt.
        gt_labels, gt_labels_counts = np.unique(gt, return_counts=True)
        pred_labels, pred_labels_counts = np.unique(pred, return_counts=True)
        gt_segms = defaultdict(dict)
        pred_segms = defaultdict(dict)
        for label, label_count in zip(gt_labels, gt_labels_counts):
            category_id = label // self.label_divisor if label > self.label_divisor else label
            gt_segms[label]['area'] = label_count
            gt_segms[label]['category_id'] = category_id
            gt_segms[label]['iscrowd'] = 1 if label in self.thing_list else 0
        for label, label_count in zip(pred_labels, pred_labels_counts):
            category_id = label // self.label_divisor if label > self.label_divisor else label
            pred_segms[label]['area'] = label_count
            pred_segms[label]['category_id'] = category_id

        # confusion matrix calculation
        pan_gt_pred = gt.astype(np.uint64) * OFFSET + pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label == self.ignore_index or pred_label == self.ignore_index:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                    'category_id']:
                continue
            union = pred_segms[pred_label]['area'] + gt_segms[gt_label][
                'area'] - intersection - gt_pred_map.get(
                    (self.ignore_index, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self.pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                self.pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false negtive
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            if gt_label == self.ignore_index:
                continue
            # ignore crowd
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            self.pq_stat[gt_info['category_id']].fn += 1

        # count false positive
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            if pred_label == self.ignore_index:
                continue
            # intersection of the segment with self.ignore_index
            intersection = gt_pred_map.get((self.ignore_index, pred_label), 0)
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get(
                    (crowd_labels_dict[pred_info['category_id']], pred_label),
                    0)
            # predicted segment is ignored if more than half of the segment correspond to self.ignore_index regions
            if intersection / pred_info['area'] > 0.5:
                continue
            self.pq_stat[pred_info['category_id']].fp += 1

    def evaluate(self):
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.pq_stat.pq_average(
                isthing=isthing, thing_list=self.thing_list)
            if name == 'All':
                results['per_class'] = per_class_results
        return OrderedDict(pan_seg=results)
