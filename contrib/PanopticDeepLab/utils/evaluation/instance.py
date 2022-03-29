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

from collections import defaultdict, OrderedDict

import numpy as np


class InstanceEvaluator(object):
    """
    Refer to 'https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py'
    Calculate the matching results of each image, each class, each IoU, and then get the final
    matching results of each class and each IoU of dataset. Base on the matching results, the AP
    and mAP can be calculated.
    we need two vectors for each class and for each overlap
    The first vector (y_true) is binary and is 1, where the ground truth says true,
    and is 0 otherwise.
    The second vector (y_score) is float [0...1] and represents the confidence of
    the prediction.
    We represent the following cases as:
                                          | y_true |   y_score
      gt instance with matched prediction |    1   | confidence
      gt instance w/o  matched prediction |    1   |     0.0
             false positive prediction    |    0   | confidence
    The current implementation makes only sense for an overlap threshold >= 0.5,
    since only then, a single prediction can either be ignored or matched, but
    never both. Further, it can never match to two gt instances.
    For matching, we vary the overlap and do the following steps:
      1.) remove all predictions that satisfy the overlap criterion with an ignore region (either void or *group)
      2.) remove matches that do not satisfy the overlap
      3.) mark non-matched predictions as false positive
    In the processing, 0 represent the first class of 'thing'. So the label will less 1 than the dataset.

    Args:
        num_classes (int): The unique number of target classes. Exclude background class, labeled 0 usually.
        overlaps (float|list, optional): The threshold of IoU. Default: 0.5.
        thing_list (list|None, optional): Thing class, only calculate AP for the thing class. Default: None.
    """

    def __init__(self, num_classes, overlaps=0.5, thing_list=None):
        super().__init__()
        self.num_classes = num_classes
        if isinstance(overlaps, float):
            overlaps = [overlaps]
        self.overlaps = overlaps
        self.y_true = [[np.empty(0) for _i in range(len(overlaps))]
                       for _j in range(num_classes)]
        self.y_score = [[np.empty(0) for _i in range(len(overlaps))]
                        for _j in range(num_classes)]
        self.hard_fns = [[0] * len(overlaps) for _ in range(num_classes)]

        if thing_list is None:
            self.thing_list = list(range(num_classes))
        else:
            self.thing_list = thing_list

    def update(self, preds, gts, ignore_mask=None):
        """
        compute y_true and y_score in this image.
        preds (list): tuple list [(label, confidence, mask), ...].
        gts (list): tuple list [(label, mask), ...].
        ignore_mask (np.ndarray): Mask to ignore.
        """

        pred_instances, gt_instances = self.get_instances(
            preds, gts, ignore_mask=ignore_mask)

        for i in range(self.num_classes):
            if i not in self.thing_list:
                continue
            for oi, oth in enumerate(self.overlaps):
                cur_true = np.ones((len(gt_instances[i])))
                cur_score = np.ones(len(gt_instances[i])) * (-float("inf"))
                cur_match = np.zeros(len(gt_instances[i]), dtype=np.bool)
                for gti, gt_instance in enumerate(gt_instances[i]):
                    found_match = False
                    for pred_instance in gt_instance['matched_pred']:
                        overlap = float(pred_instance['intersection']) / (
                            gt_instance['pixel_count'] + pred_instance[
                                'pixel_count'] - pred_instance['intersection'])
                        if overlap > oth:
                            confidence = pred_instance['confidence']

                            # if we already has a prediction for this groundtruth
                            # the prediction with the lower score is automatically a false positive
                            if cur_match[gti]:
                                max_score = max(cur_score[gti], confidence)
                                min_score = min(cur_score[gti], confidence)
                                cur_score = max_score
                                # append false positive
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, min_score)
                                cur_match = np.append(cur_match, True)
                            # otherwise set score
                            else:
                                found_match = True
                                cur_match[gti] = True
                                cur_score[gti] = confidence

                    if not found_match:
                        self.hard_fns[i][oi] += 1
                # remove not-matched ground truth instances
                cur_true = cur_true[cur_match == True]
                cur_score = cur_score[cur_match == True]

                # collect not-matched predictions as false positive
                for pred_instance in pred_instances[i]:
                    found_gt = False
                    for gt_instance in pred_instance['matched_gt']:
                        overlap = float(gt_instance['intersection']) / (
                            gt_instance['pixel_count'] + pred_instance[
                                'pixel_count'] - gt_instance['intersection'])
                        if overlap > oth:
                            found_gt = True
                            break
                    if not found_gt:
                        proportion_ignore = 0
                        if ignore_mask is not None:
                            nb_ignore_pixels = pred_instance[
                                'void_intersection']
                            proportion_ignore = float(
                                nb_ignore_pixels) / pred_instance['pixel_count']
                        if proportion_ignore <= oth:
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score,
                                                  pred_instance['confidence'])
                self.y_true[i][oi] = np.append(self.y_true[i][oi], cur_true)
                self.y_score[i][oi] = np.append(self.y_score[i][oi], cur_score)

    def evaluate(self):
        ap = self.cal_ap()
        map = self.cal_map()

        res = {}
        res["AP"] = [{i: ap[i] * 100} for i in self.thing_list]
        res["mAP"] = 100 * map

        results = OrderedDict({"ins_seg": res})
        return results

    def cal_ap(self):
        """
        calculate ap for every classes
        """
        self.ap = [0] * self.num_classes
        self.ap_overlap = [[0] * len(self.overlaps)
                           for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            if i not in self.thing_list:
                continue
            for j in range(len(self.overlaps)):
                y_true = self.y_true[i][j]
                y_score = self.y_score[i][j]
                if len(y_true) == 0:
                    self.ap_overlap[i][j] = 0
                    continue
                score_argsort = np.argsort(y_score)
                y_score_sorted = y_score[score_argsort]
                y_true_sorted = y_true[score_argsort]
                y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                # unique thresholds
                thresholds, unique_indices = np.unique(
                    y_score_sorted, return_index=True)

                # since we need to add an artificial point to the precision-recall curve
                # increase its length by 1
                nb_pr = len(unique_indices) + 1

                # calculate precision and recall
                nb_examples = len(y_score_sorted)
                nb_true_exampels = y_true_sorted_cumsum[-1]
                precision = np.zeros(nb_pr)
                recall = np.zeros(nb_pr)

                # deal with the first point
                # only thing we need to do, is to append a zero to the cumsum at the end.
                # an index of -1 uses that zero then
                y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)

                # deal with remaining
                for idx_res, idx_scores in enumerate(unique_indices):
                    cumsum = y_true_sorted_cumsum[idx_scores - 1]
                    tp = nb_true_exampels - cumsum
                    fp = nb_examples - idx_scores - tp
                    fn = cumsum + self.hard_fns[i][j]
                    p = float(tp) / (tp + fp)
                    r = float(tp) / (tp + fn)
                    precision[idx_res] = p
                    recall[idx_res] = r

                # add first point in curve
                precision[-1] = 1.
                # In some calculation，make precision the max after this point in curve.
                #precision = [np.max(precision[:i+1]) for i in range(len(precision))]
                recall[-1] = 0.

                # compute average of precision-recall curve
                # integration is performed via zero order, or equivalently step-wise integration
                # first compute the widths of each step:
                # use a convolution with appropriate kernel, manually deal with the boundaries first
                recall_for_conv = np.copy(recall)
                recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                recall_for_conv = np.append(recall_for_conv, 0.)

                step_widths = np.convolve(recall_for_conv, [-0.5, 0, 0.5],
                                          'valid')

                # integrate is now simply a dot product
                ap_current = np.dot(precision, step_widths)
                self.ap_overlap[i][j] = ap_current

        ap = [np.average(i) for i in self.ap_overlap]
        self.ap = ap

        return ap

    def cal_map(self):
        """
        calculate map for all classes
        """
        self.cal_ap()
        valid_ap = [self.ap[i] for i in self.thing_list]
        map = np.mean(valid_ap)
        self.map = map

        return map

    def get_instances(self, preds, gts, ignore_mask=None):
        """
        In this method, we create two dicts of list
        - pred_instances: contains all predictions and their associated gt
        - gtInstances:   contains all gt instances and their associated predictions

        Args:
            preds (list): Prediction of image.
            gts (list): Ground truth of image.
            ignore_mask (np.ndarray, optional): Ignore mask. Default: None.

        Return:
            dict: pred_instances, the type is dict(list(dict))), e.g. {0: [{'pred_id':0, 'label':0',
                'pixel_count':100, 'confidence': 0.9, 'void_intersection': 0,
                'matched_gt': [gt_instance0, gt_instance1, ...]}, ], 1: }
            dict: gt_instances,  the type is dict(list(dict))), e.g. {0: [{'inst_id':0, 'label':0',
                'pixel_count':100, 'mask': np.ndarray, 'matched_pred': [pred_instance0, pred_instance1, ...]}, ], 1: }
        """

        pred_instances = defaultdict(list)
        gt_instances = defaultdict(list)

        gt_inst_count = 0
        for gt in gts:
            label, mask = gt
            gt_instance = defaultdict(list)
            gt_instance['inst_id'] = gt_inst_count
            gt_instance['label'] = label
            gt_instance['pixel_count'] = np.count_nonzero(mask)
            gt_instance['mask'] = mask
            gt_instances[label].append(gt_instance)
            gt_inst_count += 1

        pred_inst_count = 0
        for pred in preds:
            label, conf, mask = pred
            pred_instance = defaultdict(list)
            pred_instance['label'] = label
            pred_instance['pred_id'] = pred_inst_count
            pred_instance['pixel_count'] = np.count_nonzero(mask)
            pred_instance['confidence'] = conf
            if ignore_mask is not None:
                pred_instance['void_intersection'] = np.count_nonzero(
                    np.logical_and(mask, ignore_mask))

            # Loop through all ground truth instances with matching label
            matched_gt = []
            for gt_num, gt_instance in enumerate(gt_instances[label]):
                # print(gt_instances)
                intersection = np.count_nonzero(
                    np.logical_and(mask, gt_instances[label][gt_num]['mask']))
                if intersection > 0:
                    gt_copy = gt_instance.copy()
                    pred_copy = pred_instance.copy()

                    gt_copy['intersection'] = intersection
                    pred_copy['intersection'] = intersection

                    matched_gt.append(gt_copy)
                    gt_instances[label][gt_num]['matched_pred'].append(
                        pred_copy)

            pred_instance['matched_gt'] = matched_gt
            pred_inst_count += 1
            pred_instances[label].append(pred_instance)

        return pred_instances, gt_instances

    @staticmethod
    def convert_gt_map(seg_map, ins_map):
        """
        Convet the ground truth with format (h*w) to the format that satisfies the AP calculation.

        Args:
            seg_map (np.ndarray): the sementic segmentation map with shape H * W. Value is 0, 1, 2, ...
            ins_map (np.ndarray): the instance segmentation map with shape H * W. Value is 0, 1, 2, ...

        Returns:
            list: tuple list like: [(label, mask), ...]
        """
        gts = []
        instance_cnt = np.unique(ins_map)
        for i in instance_cnt:
            if i == 0:
                continue
            mask = ins_map == i
            label = seg_map[mask][0]
            gts.append((label, mask.astype('int32')))
        return gts

    @staticmethod
    def convert_pred_map(seg_pred, pan_pred):
        """
        Convet the predictions with format (h*w) to the format that satisfies the AP calculation.

        Args:
            seg_pred (np.ndarray): the sementic segmentation map with shape C * H * W. Value is probability.
            pan_pred (np.ndarray): panoptic predictions, void_label, stuff_id * label_divisor, thing_id * label_divisor + ins_id , ins_id >= 1.

        Returns:
            list: tuple list like: [(label, score, mask), ...]
        """
        preds = []
        instance_cnt = np.unique(pan_pred)
        for i in instance_cnt:
            if (i < 1000) or (i % 1000 == 0):
                continue
            mask = pan_pred == i
            label = i // 1000
            score = np.mean(seg_pred[label][mask])
            preds.append((label, score, mask.astype('int32')))
        return preds
