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

import cv2
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class SemanticConnectivityLoss(nn.Layer):
    '''
    SCL (Semantic Connectivity-aware Learning) framework, which introduces a SC Loss (Semantic Connectivity-aware Loss)
    to improve the quality of segmentation results from the perspective of connectivity. Support multi-class segmentation.

    The original article refers to
        Lutao Chu, Yi Liu, Zewu Wu, Shiyu Tang, Guowei Chen, Yuying Hao, Juncai Peng, Zhiliang Yu, Zeyu Chen, Baohua Lai, Haoyi Xiong.
        "PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset"
        In WACV 2022 workshop
        https://arxiv.org/abs/2112.07146

    Running process:
    Step 1. Connected Components Calculation
    Step 2. Connected Components Matching and SC Loss Calculation
    '''

    def __init__(self, ignore_index=255, max_pred_num_conn=10, use_argmax=True):
        '''
        Args:
            ignore_index (int): Specify a pixel value to be ignored in the annotated image and does not contribute to
                the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked
                image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding
                to the original image will not be used as the independent variable of the loss function. *Default:``255``*
            max_pred_num_conn (int): Maximum number of predicted connected components. At the beginning of training,
                there will be a large number of connected components, and the calculation is very time-consuming.
                Therefore, it is necessary to limit the maximum number of predicted connected components,
                and the rest will not participate in the calculation.
            use_argmax (bool): Whether to use argmax for logits.
        '''
        super().__init__()
        self.ignore_index = ignore_index
        self.max_pred_num_conn = max_pred_num_conn
        self.use_argmax = use_argmax

    def forward(self, logits, labels):
        '''
        Args:
            logits (Tensor): [N, C, H, W]
            lables (Tensor): [N, H, W]
        '''
        preds = paddle.argmax(logits, axis=1) if self.use_argmax else logits
        preds_np = preds.astype('uint8').numpy()
        labels_np = labels.astype('uint8').numpy()
        preds = paddle.to_tensor(preds, 'float32', stop_gradient=False)
        multi_class_sc_loss = paddle.zeros([preds.shape[0]])
        zero = paddle.to_tensor([0.])  # for accelerating

        # Traverse each image
        for i in range(preds.shape[0]):
            sc_loss = 0
            class_num = 0

            pred_i = preds[i]
            preds_np_i = preds_np[i]
            labels_np_i = labels_np[i]

            # Traverse each class
            for class_ in np.unique(labels_np_i):
                if class_ == self.ignore_index:
                    continue
                class_num += 1

                # Connected Components Calculation
                preds_np_class = preds_np_i == class_
                labels_np_class = labels_np_i == class_
                pred_num_conn, pred_conn = cv2.connectedComponents(
                    preds_np_class.astype(np.uint8))  # pred_conn.shape = [H,W]
                label_num_conn, label_conn = cv2.connectedComponents(
                    labels_np_class.astype(np.uint8))

                origin_pred_num_conn = pred_num_conn
                if pred_num_conn > 2 * label_num_conn:
                    pred_num_conn = min(pred_num_conn, self.max_pred_num_conn)
                real_pred_num = pred_num_conn - 1
                real_label_num = label_num_conn - 1

                # Connected Components Matching and SC Loss Calculation
                if real_label_num > 0 and real_pred_num > 0:
                    img_connectivity = compute_class_connectiveity(
                        pred_conn, label_conn, pred_num_conn,
                        origin_pred_num_conn, label_num_conn, pred_i,
                        real_label_num, real_pred_num, zero)
                    sc_loss += 1 - img_connectivity
                elif real_label_num == 0 and real_pred_num == 0:
                    # if no connected component, SC Loss = 0, so pass
                    pass
                else:
                    preds_class = pred_i == int(class_)
                    not_preds_class = paddle.bitwise_not(preds_class)
                    labels_class = paddle.to_tensor(labels_np_class)
                    missed_detect = labels_class * not_preds_class
                    missed_detect_area = paddle.sum(missed_detect).astype(
                        'float32')
                    sc_loss += missed_detect_area / missed_detect.numel() + 1

            multi_class_sc_loss[
                i] = sc_loss / class_num if class_num != 0 else 0
        multi_class_sc_loss = paddle.mean(multi_class_sc_loss)
        return multi_class_sc_loss


def compute_class_connectiveity(pred_conn, label_conn, pred_num_conn,
                                origin_pred_num_conn, label_num_conn, pred,
                                real_label_num, real_pred_num, zero):

    pred_conn = paddle.to_tensor(pred_conn)
    label_conn = paddle.to_tensor(label_conn)
    pred_conn = F.one_hot(pred_conn, origin_pred_num_conn)
    label_conn = F.one_hot(label_conn, label_num_conn)

    ious = paddle.zeros((real_label_num, real_pred_num))
    pair_conn_sum = paddle.to_tensor([0.], stop_gradient=False)

    for i in range(1, label_num_conn):
        label_i = label_conn[:, :, i]

        pair_conn = paddle.to_tensor([0.], stop_gradient=False)
        pair_conn_num = 0

        for j in range(1, pred_num_conn):
            pred_j_mask = pred_conn[:, :, j]
            pred_j = pred_j_mask * pred

            iou = compute_iou(pred_j, label_i, zero)
            ious[i - 1, j - 1] = iou
            if iou != 0:
                pair_conn += iou
                pair_conn_num += 1

        if pair_conn_num != 0:
            pair_conn_sum += pair_conn / pair_conn_num
    lone_pred_num = 0

    pred_sum = paddle.sum(ious, axis=0)
    for m in range(0, real_pred_num):
        if pred_sum[m] == 0:
            lone_pred_num += 1
    img_connectivity = pair_conn_sum / (real_label_num + lone_pred_num)
    return img_connectivity


def compute_iou(pred_i, label_i, zero):
    intersect_area_i = paddle.sum(pred_i * label_i)
    if paddle.equal(intersect_area_i, zero):
        return 0

    pred_area_i = paddle.sum(pred_i)
    label_area_i = paddle.sum(label_i)
    union_area_i = pred_area_i + label_area_i - intersect_area_i
    if paddle.equal(union_area_i, zero):
        return 1
    else:
        return intersect_area_i / union_area_i
