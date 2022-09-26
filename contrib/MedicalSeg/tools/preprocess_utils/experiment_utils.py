# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from copy import deepcopy


def pad_shape(shape, must_be_divisible_by):
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shape = [
        shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i]
        for i in range(len(shape))
    ]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shape[i] -= must_be_divisible_by[i]
    new_shape = np.array(new_shape).astype(int)
    return new_shape


def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2**np.array(net_numpool_per_axis)


def get_network_numpool(patch_size, maxpool_cap=999, min_feature_map_size=4):
    network_numpool_per_axis = np.floor(
        [np.log(i / min_feature_map_size) / np.log(2)
         for i in patch_size]).astype(int)
    network_numpool_per_axis = [
        min(i, maxpool_cap) for i in network_numpool_per_axis
    ]
    return network_numpool_per_axis


def get_pool_and_conv_props_poolLateV2(patch_size, min_feature_map_size,
                                       max_numpool, spacing):
    initial_spacing = deepcopy(spacing)
    reach = max(initial_spacing)
    dim = len(patch_size)

    num_pool_per_axis = get_network_numpool(patch_size, max_numpool,
                                            min_feature_map_size)

    net_num_pool_op_kernel_sizes = []
    net_conv_kernel_sizes = []
    net_numpool = max(num_pool_per_axis)

    current_spacing = spacing
    for p in range(net_numpool):
        reached = [current_spacing[i] / reach > 0.5 for i in range(dim)]
        pool = [
            2 if num_pool_per_axis[i] + p >= net_numpool else 1
            for i in range(dim)
        ]
        if all(reached):
            conv = [3] * dim
        else:
            conv = [3 if not reached[i] else 1 for i in range(dim)]
        net_num_pool_op_kernel_sizes.append(pool)
        net_conv_kernel_sizes.append(conv)
        current_spacing = [i * j for i, j in zip(current_spacing, pool)]

    net_conv_kernel_sizes.append([3] * dim)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    return num_pool_per_axis, net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, patch_size, must_be_divisible_by


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size,
                            max_numpool):
    dim = len(spacing)
    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [
            i for i in range(dim) if current_spacing[i] / min_spacing < 2
        ]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [
                i for i in range(dim)
                if current_spacing[i] / my_spacing < 2 and my_spacing /
                current_spacing[i] < 2
            ]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        valid_axes_for_pool = [
            i for i in valid_axes_for_pool
            if current_size[i] >= 2 * min_feature_map_size
        ]

        valid_axes_for_pool = [
            i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool
        ]

        if len(valid_axes_for_pool) == 0:
            break

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    conv_kernel_sizes.append([3] * dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by


def compute_approx_vram_consumption(patch_size,
                                    num_pool_per_axis,
                                    base_num_features,
                                    max_num_features,
                                    num_modalities,
                                    num_classes,
                                    pool_op_kernel_sizes,
                                    deep_supervision=False,
                                    conv_per_stage=2):
    if not isinstance(num_pool_per_axis, np.ndarray):
        num_pool_per_axis = np.array(num_pool_per_axis)

    npool = len(pool_op_kernel_sizes)

    map_size = np.array(patch_size)
    tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(
        map_size, dtype=np.int64) * base_num_features + num_modalities *
                   np.prod(
                       map_size, dtype=np.int64) + num_classes * np.prod(
                           map_size, dtype=np.int64))

    num_feat = base_num_features

    for p in range(npool):
        for pi in range(len(num_pool_per_axis)):
            map_size[pi] /= pool_op_kernel_sizes[p][pi]
        num_feat = min(num_feat * 2, max_num_features)
        num_blocks = (conv_per_stage * 2 + 1) if p < (
            npool - 1
        ) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
        tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
        if deep_supervision and p < (npool - 2):
            tmp += np.prod(map_size, dtype=np.int64) * num_classes
    return tmp
