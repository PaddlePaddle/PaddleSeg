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

import numpy as np
import cv2
import os


def get_round(data):
    round = 0.5 if data >= 0 else -0.5
    return (int)(data + round)


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """计算光流跟踪匹配点和光流图
    输入参数:
        pre_gray: 上一帧灰度图
        cur_gray: 当前帧灰度图
        prev_cfd: 上一帧光流图
        dl_weights: 融合权重图
        disflow: 光流数据结构
    返回值:
        is_track: 光流点跟踪二值图，即是否具有光流点匹配
        track_cfd: 光流跟踪图
    """
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    for r in range(h):
        for c in range(w):
            fxy_fw = flow_fw[r, c]
            dx_fw = get_round(fxy_fw[0])
            cur_x = dx_fw + c
            dy_fw = get_round(fxy_fw[1])
            cur_y = dy_fw + r
            if cur_x < 0 or cur_x >= w or cur_y < 0 or cur_y >= h:
                continue
            fxy_bw = flow_bw[cur_y, cur_x]
            dx_bw = get_round(fxy_bw[0])
            dy_bw = get_round(fxy_bw[1])
            if ((dy_fw + dy_bw) * (dy_fw + dy_bw) +
                (dx_fw + dx_bw) * (dx_fw + dx_bw)) >= check_thres:
                continue
            if abs(dy_fw) <= 0 and abs(dx_fw) <= 0 and abs(dy_bw) <= 0 and abs(
                    dx_bw) <= 0:
                dl_weights[cur_y, cur_x] = 0.05
            is_track[cur_y, cur_x] = 1
            track_cfd[cur_y, cur_x] = prev_cfd[r, c]
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    fusion_cfd = dl_cfd.copy()
    idxs = np.where(is_track > 0)
    for i in range(len(idxs[0])):
        x, y = idxs[0][i], idxs[1][i]
        dl_score = dl_cfd[x, y]
        track_score = track_cfd[x, y]
        fusion_cfd[x, y] = dl_weights[x, y] * dl_score + (
            1 - dl_weights[x, y]) * track_score
        if dl_score > 0.9 or dl_score < 0.1:
            if dl_weights[x, y] < 0.1:
                fusion_cfd[x, y] = 0.3 * dl_score + 0.7 * track_score
            else:
                fusion_cfd[x, y] = 0.4 * dl_score + 0.6 * track_score
        else:
            fusion_cfd[x, y] = dl_weights[x, y] * dl_score + (
                1 - dl_weights[x, y]) * track_score
    return fusion_cfd


def postprocess(cur_gray, scoremap, prev_gray, pre_cfd, disflow, is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        pre_gray : 前一帧灰度图
        pre_cfd  ：前一帧融合结果
        scoremap : 当前帧分割结果
        difflow  : 光流
        is_init : 是否第一帧
    Returns:
        fusion_cfd : 光流追踪图和预测结果融合图
    """
    height, width = scoremap.shape[0], scoremap.shape[1]
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    h, w = scoremap.shape
    cur_cfd = scoremap.copy()

    if is_init:
        is_init = False
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((w, h), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, pre_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)

    fusion_cfd = cv2.GaussianBlur(fusion_cfd, (3, 3), 0)

    return fusion_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)
