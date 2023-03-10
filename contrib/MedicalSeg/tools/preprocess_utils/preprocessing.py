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
import pickle
import os
import numpy as np
from collections import OrderedDict
from scipy.ndimage.interpolation import map_coordinates
from multiprocessing.pool import Pool
from skimage.transform import resize
from .image_crop import get_case_identifier_from_npz, ImageCropper
from .path_utils import join_paths


def resize_segmentation(segmentation, new_shape, order=3):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(
        new_shape
    ), "New shape must have same dimensionality as segmentation, but got {} and {}.".format(
        segmentation.shape, new_shape)
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(
                mask.astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis


def resample_patient(data,
                     seg,
                     original_spacing,
                     target_spacing,
                     order_data=3,
                     order_seg=0,
                     force_separate_z=False,
                     order_z_data=0,
                     order_z_seg=0,
                     separate_z_anisotropy_threshold=3):
    assert not ((data is None) and (seg is None))
    if data is not None:
        assert len(data.shape) == 4, "Data geometry must be c x y z."
    if seg is not None:
        assert len(seg.shape) == 4, "Segmentation geometry must be c x y z."

    if data is not None:
        shape = np.array(data[0].shape)
    else:
        shape = np.array(seg[0].shape)
    new_shape = np.round(
        ((np.array(original_spacing) / np.array(target_spacing)).astype(float) *
         shape)).astype(int)

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(original_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(original_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(original_spacing)
        elif get_do_separate_z(target_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(target_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
        elif len(axis) == 2:
            do_separate_z = False
        else:
            pass

    if data is not None:
        data_reshaped = resample_data_or_seg(
            data,
            new_shape,
            False,
            axis,
            order_data,
            do_separate_z,
            order_z=order_z_data)
    else:
        data_reshaped = None
    if seg is not None:
        seg_reshaped = resample_data_or_seg(
            seg,
            new_shape,
            True,
            axis,
            order_seg,
            do_separate_z,
            order_z=order_z_seg)
    else:
        seg_reshaped = None
    return data_reshaped, seg_reshaped


def resample_data_or_seg(data,
                         new_shape,
                         is_seg,
                         axis=None,
                         order=3,
                         do_separate_z=False,
                         order_z=0):
    assert len(data.shape) == 4, "Data geometry must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            print("separate z, order in z is {} order inplane is {}.".format(
                order_z, order))
            assert len(
                axis
            ) == 1, "Only one anisotropic axis supported, but got {} axis.".format(
                len(axis))
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(
                            resize_fn(data[c, slice_id], new_shape_2d, order, **
                                      kwargs))
                    elif axis == 1:
                        reshaped_data.append(
                            resize_fn(data[c, :, slice_id], new_shape_2d, order,
                                      **kwargs))
                    else:
                        reshaped_data.append(
                            resize_fn(data[c, :, :, slice_id], new_shape_2d,
                                      order, **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(
                            map_coordinates(
                                reshaped_data,
                                coord_map,
                                order=order_z,
                                mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates(
                                    (reshaped_data == cl).astype(float),
                                    coord_map,
                                    order=order_z,
                                    mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order {}.".format(order))
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(
                    resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("No resampling necessary.")
        return data


class GenericPreprocessor:
    def __init__(self,
                 normalization_scheme_per_modality,
                 use_nonzero_mask,
                 transpose_forward: (tuple, list),
                 intensityproperties=None):
        self.transpose_forward = transpose_forward
        self.intensityproperties = intensityproperties
        self.normalization_scheme_per_modality = normalization_scheme_per_modality
        self.use_nonzero_mask = use_nonzero_mask

        self.resample_separate_z_anisotropy_threshold = 3

    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(
            join_paths(cropped_output_dir, "%s.npz" % case_identifier))['data']
        data = all_data[:-1].astype(np.float32)
        seg = all_data[-1:]
        with open(
                join_paths(cropped_output_dir, "%s.pkl" % case_identifier),
                'rb') as f:
            properties = pickle.load(f)
        return data, seg, properties

    def resample_and_normalize(self,
                               data,
                               target_spacing,
                               properties,
                               seg=None,
                               force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[
            self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        data[np.isnan(data)] = 0

        data, seg = resample_patient(
            data,
            seg,
            np.array(original_spacing_transposed),
            target_spacing,
            3,
            1,
            force_separate_z=force_separate_z,
            order_z_data=0,
            order_z_seg=0,
            separate_z_anisotropy_threshold=self.
            resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("Resample before: {}".format(before))
        print("Resample after: {}: ".format(after))

        if seg is not None:
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(
            data
        ), "normalization_scheme_per_modality must have as many entries as data has modalities in GenericPreprocessor."
        assert len(self.use_nonzero_mask) == len(
            data
        ), "use_nonzero_mask must have as many entries as data has modalities in GenericPreprocessor."
        print("Normalize...")
        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                assert self.intensityproperties is not None, "CT dataset must has intensityproperties, but got None. Please check your DataAnalyzer."
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                assert self.intensityproperties is not None, "CT dataset must has intensityproperties, but got None. Please check your DataAnalyzer."
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mean_intensity = data[c][mask].mean()
                std_intensity = data[c][mask].std()
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                    data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (
                        data[c][mask].std() + 1e-8)
                    data[c][mask == 0] = 0
                else:
                    mn = data[c].mean()
                    std = data[c].std()
                    data[c] = (data[c] - mn) / (std + 1e-8)
        print("Normalize Done.")
        return data, seg, properties

    def preprocess_test_case(self,
                             data_files,
                             target_spacing,
                             seg_file=None,
                             force_separate_z=None):
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_files,
                                                                     seg_file)

        data = data.transpose((0, * [i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, * [i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(
            data,
            target_spacing,
            properties,
            seg,
            force_separate_z=force_separate_z)
        return data.astype(np.float32), seg, properties

    def _run_internal(self, target_spacing, case_identifier,
                      output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes):
        data, seg, properties = self.load_cropped(cropped_output_dir,
                                                  case_identifier)

        data = data.transpose((0, * [i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, * [i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(
            data, target_spacing, properties, seg, force_separate_z)

        all_data = np.vstack((data, seg)).astype(np.float32)

        num_samples = 10000
        min_percent_coverage = 0.01
        rndst = np.random.RandomState(1234)
        class_locs = {}
        for c in all_classes:
            all_locs = np.argwhere(all_data[-1] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(
                target_num_samples,
                int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(
                len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
        properties['class_locations'] = class_locs

        print("saving: ",
              join_paths(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(
            join_paths(output_folder_stage, "%s.npz" % case_identifier),
            data=all_data.astype(np.float32))
        with open(
                join_paths(output_folder_stage, "%s.pkl" % case_identifier),
                'wb') as f:
            pickle.dump(properties, f)

    def run(self,
            target_spacings,
            input_folder_with_cropped_npz,
            output_folder,
            data_identifier,
            num_threads=1,
            force_separate_z=None):
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = [
            join_paths(input_folder_with_cropped_npz, file_name)
            for file_name in os.listdir(input_folder_with_cropped_npz)
            if os.path.isfile(
                join_paths(input_folder_with_cropped_npz, file_name)) and
            file_name.endswith('.npz')
        ]
        list_of_cropped_npz_files.sort()
        os.makedirs(output_folder, exist_ok=True)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages
        with open(
                join_paths(input_folder_with_cropped_npz,
                           'dataset_properties.pkl'), 'rb') as f:
            all_classes = pickle.load(f)['all_classes']

        for i in range(num_stages):
            all_args = []
            output_folder_stage = join_paths(output_folder,
                                             data_identifier + "_stage%d" % i)
            os.makedirs(output_folder_stage, exist_ok=True)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                all_args.append(args)
            p = Pool(num_threads[i])
            p.starmap(self._run_internal, all_args)
            p.close()
            p.join()


class PreprocessorFor2D(GenericPreprocessor):
    def __init__(self,
                 normalization_scheme_per_modality,
                 use_nonzero_mask,
                 transpose_forward: (tuple, list),
                 intensityproperties=None):
        super(PreprocessorFor2D, self).__init__(
            normalization_scheme_per_modality, use_nonzero_mask,
            transpose_forward, intensityproperties)

    def run(self,
            target_spacings,
            input_folder_with_cropped_npz,
            output_folder,
            data_identifier,
            num_threads=1,
            force_separate_z=None):
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = [
            join_paths(input_folder_with_cropped_npz, file_name)
            for file_name in os.listdir(input_folder_with_cropped_npz)
            if os.path.isfile(
                join_paths(input_folder_with_cropped_npz, file_name)) and
            file_name.endswith('.npz')
        ]
        list_of_cropped_npz_files.sort()
        assert len(list_of_cropped_npz_files) != 0, "set list of files first"
        os.makedirs(output_folder, exist_ok=True)
        all_args = []
        num_stages = len(target_spacings)

        with open(
                join_paths(input_folder_with_cropped_npz,
                           'dataset_properties.pkl'), 'rb') as f:
            all_classes = pickle.load(f)['all_classes']

        for i in range(num_stages):
            output_folder_stage = join_paths(output_folder,
                                             data_identifier + "_stage%d" % i)
            os.makedirs(output_folder_stage, exist_ok=True)
            spacing = target_spacings[i]
            for j, case in enumerate(list_of_cropped_npz_files):
                case_identifier = get_case_identifier_from_npz(case)
                args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                all_args.append(args)
        p = Pool(num_threads)
        p.starmap(self._run_internal, all_args)
        p.close()
        p.join()

    def resample_and_normalize(self,
                               data,
                               target_spacing,
                               properties,
                               seg=None,
                               force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[
            self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(
            data,
            seg,
            np.array(original_spacing_transposed),
            target_spacing,
            3,
            1,
            force_separate_z=force_separate_z,
            order_z_data=0,
            order_z_seg=0,
            separate_z_anisotropy_threshold=self.
            resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("Resample before: {}".format(before))
        print("Resample after: {}: ".format(after))

        if seg is not None:
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        assert len(self.normalization_scheme_per_modality) == len(
            data
        ), "normalization_scheme_per_modality must have as many entries as data has modalities in GenericPreprocessor."
        assert len(self.use_nonzero_mask) == len(
            data
        ), "use_nonzero_mask must have as many entries as data has modalities in GenericPreprocessor."
        print("Normalize...")
        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                assert self.intensityproperties is not None, "CT dataset must has intensityproperties, but got None. Please check your DataAnalyzer."
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                assert self.intensityproperties is not None, "CT dataset must has intensityproperties, but got None. Please check your DataAnalyzer."
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if use_nonzero_mask[c]:
                    mask = seg[-1] >= 0
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (
                    data[c][mask].std() + 1e-8)
                data[c][mask == 0] = 0
        print("Normalization done")
        return data, seg, properties
