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

from collections import defaultdict

import numpy as np

from paddlepanseg.cvlibs import build_info_dict
from paddlepanseg.utils import decode_pan_id
from .evaluator import Evaluator

OFFSET = 256 * 256 * 256


class PQStatCat(object):
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self

    def __repr__(self):
        s = 'iou: ' + str(self.iou) + ' tp: ' + str(self.tp) + ' fp: ' + str(
            self.fp) + ' fn: ' + str(self.fn)
        return s


class PQStat(object):
    def __init__(self, all_ids):
        self.pq_per_cat = defaultdict(PQStatCat)
        self.all_ids = all_ids

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, isthing=None, thing_ids=None):
        """
        Calculate the average pq for all and every class.
        
        Args:
            isthing (bool|None, optional): calculate average pq for thing class if isthing is True,
                for stuff class if isthing is False and for all if isthing is None. Default: None. Default: None.
            thing_ids (list|None, optional): A list of thing class. It should be provided when isthing is equal to True or False. Default: None.
        """
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label in self.all_ids:
            if isthing is not None:
                if isthing:
                    if label not in thing_ids:
                        continue
                else:
                    if label in thing_ids:
                        continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = 100 * iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = 100 * iou / tp if tp != 0 else 0
            rq_class = 100 * tp / (tp + 0.5 * fp + 0.5 * fn)
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


class PanSegEvaluator(Evaluator):
    def __init__(self,
                 num_classes,
                 thing_ids,
                 label_divisor=1000,
                 convert_id=None):
        self.num_classes = num_classes
        self.all_ids = list(range(0, self.num_classes))
        self.thing_ids = list(thing_ids)
        self.label_divisor = label_divisor
        self.convert_id = convert_id
        if self.convert_id is not None:
            self.thing_ids = [self.convert_id(id_) for id_ in self.thing_ids]
            self.all_ids = [self.convert_id(id_) for id_ in self.all_ids]
        self.pq_stat = PQStat(self.all_ids)

    def update(self, sample_dict, pp_out_dict):
        VOID = 0  # id==0 stands for area to be ignored

        pred = pp_out_dict['pan_pred'].squeeze().numpy()
        gt = sample_dict['pan_label'].squeeze().numpy()
        # Use sample['ann'], rather than parsing info from gt.
        # Note that the consistency between gt and gt_ann must be ensured.
        gt_ann = sample_dict['ann'][0]
        if self.convert_id is not None:
            for el in gt_ann:
                el['category_id'] = self.convert_id(el['category_id'])

        pred_ann = []
        labels, labels_cnt = np.unique(pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label == VOID:
                continue
            cat_id, _ = decode_pan_id(label, self.label_divisor)
            if self.convert_id is not None:
                cat_id = self.convert_id(cat_id)
            pred_ann.append({
                'id': label,
                'category_id': cat_id,
                'area': label_cnt
            })

        gt_segms = {el['id']: el for el in gt_ann}
        pred_segms = {el['id']: el for el in pred_ann}

        # Calculate confusion matrix
        pan_gt_pred = gt.astype(np.uint64) * OFFSET + pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # Count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label][
                    'category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - \
                    gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                self.pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                self.pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # Ignore crowd seegments
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            self.pq_stat[gt_info['category_id']].fn += 1

        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            intersection = gt_pred_map.get((VOID, pred_label), 0)

            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((
                    crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            if intersection / pred_info['area'] > 0.5:
                continue
            self.pq_stat[pred_info['category_id']].fp += 1

    def evaluate(self):
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.pq_stat.pq_average(
                isthing=isthing, thing_ids=self.thing_ids)
            if name == 'All':
                results['per_class'] = per_class_results
        return build_info_dict(_type_='metric', pan_metrics=results)
