# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

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

import os
import pickle
import numpy as np
from collections import OrderedDict
from abc import abstractmethod


class SlimDataLoaderBase(object):
    def __init__(self,
                 data,
                 batch_size,
                 number_of_threads_in_multithreaded=None):
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        raise NotImplementedError


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self,
                 data,
                 patch_size,
                 final_patch_size,
                 batch_size,
                 has_prev_stage=False,
                 oversample_foreground_percent=0.0,
                 memmap_mode="r",
                 pad_mode="edge",
                 pad_kwargs_data=None,
                 pad_sides=None):
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())

        self.need_to_pad = (
            np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size *
                                     (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if os.path.isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy",
                                    self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size,
                                         True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                with open(self._data[i]['properties_file'], 'rb') as f:
                    properties = pickle.load(f)
            case_properties.append(properties)

            if os.path.isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(
                    self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            if self.has_prev_stage:
                if not os.path.isfile(self._data[i][
                        'seg_from_prev_stage_file']):
                    raise UserWarning(
                        "seg from prev stage missing: {}. please run single_fold_eval.py with --predict_next_stage first.".
                        format(self._data[i]['seg_from_prev_stage_file']))
                if os.path.isfile(self._data[i]['seg_from_prev_stage_file'][:-4]
                                  + ".npy"):
                    segs_from_previous_stage = np.load(
                        self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                        mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i][
                        'seg_from_prev_stage_file'])['data'][None]

                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:
                                                                   seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                if need_to_pad[d] + case_all_data.shape[d +
                                                        1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[
                        d + 1]

            shape = case_all_data.shape[1:]
            lb_x = -need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[
                0] % 2 - self.patch_size[0]
            lb_y = -need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[
                1] % 2 - self.patch_size[1]
            lb_z = -need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[
                2] % 2 - self.patch_size[2]

            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                if 'class_locations' not in properties.keys():
                    raise RuntimeError(
                        "Please rerun the preprocessing with the newest version of nnU-Net!"
                    )

                foreground_classes = np.array([
                    i for i in properties['class_locations'].keys()
                    if len(properties['class_locations'][i]) != 0
                ])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)
                    voxels_of_that_class = properties['class_locations'][
                        selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(
                        len(voxels_of_that_class))]
                    bbox_x_lb = max(lb_x,
                                    selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y,
                                    selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z,
                                    selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = np.copy(
                case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                              valid_bbox_y_lb:valid_bbox_y_ub, valid_bbox_z_lb:
                              valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:,
                                                                  valid_bbox_x_lb:
                                                                  valid_bbox_x_ub,
                                                                  valid_bbox_y_lb:
                                                                  valid_bbox_y_ub,
                                                                  valid_bbox_z_lb:
                                                                  valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], (
                (0, 0), (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], (
                (0, 0), (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))), 'constant',
                               **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, (
                    (0, 0), (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)), (
                        -min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})
        return {
            'data': data,
            'seg': seg,
            'properties': case_properties,
            'keys': selected_keys
        }


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self,
                 data,
                 patch_size,
                 final_patch_size,
                 batch_size,
                 oversample_foreground_percent=0.0,
                 memmap_mode="r",
                 pseudo_3d_slices=1,
                 pad_mode="edge",
                 pad_kwargs_data=None,
                 pad_sides=None):
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1
        k = list(self._data.keys())[0]
        if os.path.isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy",
                                    self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size *
                                     (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size,
                                         True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                with open(self._data[i]['properties_file'], 'rb') as f:
                    properties = pickle.load(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not os.path.isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] +
                                        ".npz")['data']
            else:
                case_all_data = np.load(
                    self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                if 'class_locations' not in properties.keys():
                    raise RuntimeError(
                        "Please rerun the preprocessing with the newest version of nnU-Net!"
                    )

                foreground_classes = np.array([
                    i for i in properties['class_locations'].keys()
                    if len(properties['class_locations'][i]) != 0
                ])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][
                        selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[
                        voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate(
                        (np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate(
                        (case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape(
                    (-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            assert len(
                case_all_data.shape
            ) == 3, "case data shape shoud be (c, x, y), but got {}.".format(
                case_all_data.shape)

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                if need_to_pad[d] + case_all_data.shape[d +
                                                        1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[
                        d + 1]

            shape = case_all_data.shape[1:]
            lb_x = -need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[
                0] % 2 - self.patch_size[0]
            lb_y = -need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[
                1] % 2 - self.patch_size[1]

            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                selected_voxel = voxels_of_that_class[np.random.choice(
                    len(voxels_of_that_class))]
                bbox_x_lb = max(lb_x,
                                selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y,
                                selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], (
                (0, 0), (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], (
                (0, 0), (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))), 'constant',
                                           **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {
            'data': data,
            'seg': seg,
            'properties': case_properties,
            "keys": keys
        }
