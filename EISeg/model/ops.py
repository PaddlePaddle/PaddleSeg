# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/saic-vul/ritm_interactive_segmentation
Ths copyright of saic-vul/ritm_interactive_segmentation is as follows:
MIT License [see LICENSE for details]
"""


import paddle
import paddle.nn as nn
import numpy as np


class DistMaps(nn.Layer):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=True, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks

        if self.cpu_mode:
            from util.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].numpy().astype('float32'), rows, cols,
                                                  norm_delimeter))
            coords = paddle.to_tensor(np.stack(coords, axis=0)).astype('float32')
        else:
            num_points = points.shape[1] // 2
            points = points.reshape([-1, points.shape[2]])
            points, points_order = paddle.split(points, [2, 1], axis=1)
            invalid_points = paddle.max(points, axis=1, keepdim=False) < 0
            row_array = paddle.arange(start=0, end=rows, step=1, dtype='float32')
            col_array = paddle.arange(start=0, end=cols, step=1, dtype='float32')

            coord_rows, coord_cols = paddle.meshgrid(row_array, col_array)
            coords = paddle.unsqueeze(paddle.stack([coord_rows, coord_cols], axis=0), axis=0).tile(
                [points.shape[0], 1, 1, 1])
            
            add_xy = (points * self.spatial_scale).reshape([points.shape[0], points.shape[1], 1, 1])
            coords = coords - add_xy
            if not self.use_disks:
                coords = coords / (self.norm_radius * self.spatial_scale)

            coords = coords * coords
            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]
            invalid_points = invalid_points.numpy()

            coords[invalid_points, :, :, :] = 1e6
            coords = coords.reshape([-1, num_points, 1, rows, cols])
            coords = paddle.min(coords, axis=1)
            coords = coords.reshape([-1, 2, rows, cols])

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).astype('float32')
        else:
            coords = paddle.tanh(paddle.sqrt(coords) * 2)
        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class ScaleLayer(nn.Layer):
    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = self.create_parameter(shape=[1],
                                           dtype='float32',
                                           default_initializer=nn.initializer.Constant(init_value / lr_mult))

    def forward(self, x):
        scale = paddle.abs(self.scale * self.lr_mult)
        return x * scale


class BatchImageNormalize:
    def __init__(self, mean, std):
        self.mean = paddle.to_tensor(np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis]).astype('float32')
        self.std = paddle.to_tensor(np.array(std)[np.newaxis, :,np.newaxis, np.newaxis]).astype('float32')

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / self.std
        return tensor
