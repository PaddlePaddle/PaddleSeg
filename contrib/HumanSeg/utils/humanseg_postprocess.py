# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
# ==============================================================================

import os
import numpy as np
import cv2


def humanseg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
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
    hgt, wdh = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    # 计算前向光流
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    # 计算后向光流
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    get_round = lambda data: (int)(data + 0.5) if data >= 0 else (int)(data -
                                                                       0.5)
    for row in range(hgt):
        for col in range(wdh):
            # 计算光流处理后对应点坐标
            # (row, col) -> (cur_x, cur_y)
            fxy_fw = flow_fw[row, col]
            dx_fw = get_round(fxy_fw[0])
            cur_x = dx_fw + col
            dy_fw = get_round(fxy_fw[1])
            cur_y = dy_fw + row
            if cur_x < 0 or cur_x >= wdh or cur_y < 0 or cur_y >= hgt:
                continue
            fxy_bw = flow_bw[cur_y, cur_x]
            dx_bw = get_round(fxy_bw[0])
            dy_bw = get_round(fxy_bw[1])
            # 光流移动小于阈值
            lmt = ((dy_fw + dy_bw) * (dy_fw + dy_bw) +
                   (dx_fw + dx_bw) * (dx_fw + dx_bw))
            if lmt >= check_thres:
                continue
            # 静止点降权
            if abs(dy_fw) <= 0 and abs(dx_fw) <= 0 and abs(dy_bw) <= 0 and abs(
                    dx_bw) <= 0:
                dl_weights[cur_y, cur_x] = 0.05
            is_track[cur_y, cur_x] = 1
            track_cfd[cur_y, cur_x] = prev_cfd[row, col]
    return track_cfd, is_track, dl_weights


def humanseg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回值:
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    cur_cfd = dl_cfd.copy()
    idxs = np.where(is_track > 0)
    for i in range(len(idxs)):
        x, y = idxs[0][i], idxs[1][i]
        dl_score = dl_cfd[x, y]
        track_score = track_cfd[x, y]
        if dl_score > 0.9 or dl_score < 0.1:
            if dl_weights[x, y] < 0.1:
                cur_cfd[x, y] = 0.3 * dl_score + 0.7 * track_score
            else:
                cur_cfd[x, y] = 0.4 * dl_score + 0.6 * track_score
        else:
            cur_cfd[x, y] = dl_weights[x, y] * dl_score + (
                1 - dl_weights[x, y]) * track_score
    return cur_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    """设置背景和前景阈值mask
    输入参数:
        img : 原始图像, np.uint8 类型.
        thresh_bg : 背景阈值百分比，低于该值置为0.
        thresh_fg : 前景阈值百分比，超过该值置为1.
    返回值:
        dst : 原始图像设置完前景背景阈值mask结果, np.float32 类型.
    """
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optflow_handle(cur_gray, scoremap, is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        scoremap : 当前帧分割结果
        is_init : 是否第一帧
    Returns:
        dst : 光流追踪图和预测结果融合图, 类型为 np.float32
    """
    height, width = scoremap.shape[0], scoremap.shape[1]
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev_gray = np.zeros((height, width), np.uint8)
    prev_cfd = np.zeros((height, width), np.float32)
    cur_cfd = scoremap.copy()
    if is_init:
        is_init = False
        if height <= 64 or width <= 64:
            disflow.setFinestScale(1)
        elif height <= 160 or width <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((height, width), np.float32) * 0.3
        track_cfd, is_track, weights = humanseg_tracking(
            prev_gray, cur_gray, prev_cfd, weights, disflow)
        fusion_cfd = humanseg_track_fuse(track_cfd, cur_cfd, weights, is_track)
    fusion_cfd = cv2.GaussianBlur(fusion_cfd, (3, 3), 0)
    return fusion_cfd


def postprocess(image, output_data):
    """对预测结果进行后处理
    Args:
         image: 原始图，opencv 图片对象
         output_data: Paddle预测结果原始数据
    Returns:
         原图和预测结果融合并做了光流优化的结果图
    """
    scoremap = output_data[:, :, 1]
    scoremap = (scoremap * 255).astype(np.uint8)
    # 光流处理
    cur_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    optflow_map = optflow_handle(cur_gray, scoremap, False)
    optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
    optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
    optflow_map = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
    return optflow_map
