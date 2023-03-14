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
import shutil
from collections import OrderedDict

from .image_crop import get_case_identifier_from_npz
from .preprocessing import GenericPreprocessor, PreprocessorFor2D
from .experiment_utils import *
from .path_utils import join_paths

DEFAULT_BATCH_SIZE_3D = 2
DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
SPACING_FACTOR_BETWEEN_STAGES = 2
BASE_NUM_FEATURES_3D = 30
MAX_NUMPOOL_3D = 999
MAX_NUM_FILTERS_3D = 320

DEFAULT_PATCH_SIZE_2D = (256, 256)
BASE_NUM_FEATURES_2D = 30
DEFAULT_BATCH_SIZE_2D = 50
MAX_NUMPOOL_2D = 999
MAX_FILTERS_2D = 480

use_this_for_batch_size_computation_2D = 19739648
use_this_for_batch_size_computation_3D = 520000000


class ExperimentPlanner:
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        os.makedirs(preprocessed_output_folder, exist_ok=True)
        self.folder_with_cropped_data = folder_with_cropped_data
        self.preprocessed_output_folder = preprocessed_output_folder
        self.list_of_cropped_npz_files = [
            join_paths(self.folder_with_cropped_data, file_name)
            for file_name in os.listdir(self.folder_with_cropped_data)
            if file_name.endswith('.npz') and os.path.isfile(
                join_paths(self.folder_with_cropped_data, file_name))
        ]
        self.list_of_cropped_npz_files.sort()

        self.preprocessor_name = GenericPreprocessor

        assert os.path.isfile(
            join_paths(self.folder_with_cropped_data, "dataset_properties.pkl")
        ), "folder_with_cropped_data must contain dataset_properties.pkl"
        with open(
                join_paths(self.folder_with_cropped_data,
                           "dataset_properties.pkl"), 'rb') as f:
            self.dataset_properties = pickle.load(f)

        self.plans_per_stage = OrderedDict()
        self.plans = OrderedDict()

        self.transpose_forward = [0, 1, 2]
        self.transpose_backward = [0, 1, 2]

        self.unet_base_num_features = BASE_NUM_FEATURES_3D
        self.unet_max_num_filters = 320
        self.unet_max_numpool = 999
        self.unet_min_batch_size = 2
        self.unet_featuremap_min_edge_length = 4

        self.target_spacing_percentile = 50
        self.anisotropy_threshold = 3
        self.how_much_of_a_patient_must_the_network_see_at_stage0 = 4
        self.batch_size_covers_max_percent_of_dataset = 0.05
        self.conv_per_stage = 2

    def get_target_spacing(self):
        spacings = self.dataset_properties['all_spacings']
        target = np.percentile(
            np.vstack(spacings), self.target_spacing_percentile, 0)
        return target

    def save_my_plans(self):
        with open(self.plans_fname, 'wb') as f:
            pickle.dump(self.plans, f)

    def load_my_plans(self):
        with open(self.plans_fname, 'rb') as f:
            self.plans = pickle.load(f)

        self.plans_per_stage = self.plans['plans_per_stage']
        self.dataset_properties = self.plans['dataset_properties']

        self.transpose_forward = self.plans['transpose_forward']
        self.transpose_backward = self.plans['transpose_backward']

    def get_properties_for_stage(self, current_spacing, original_spacing,
                                 original_shape, num_cases, num_modalities,
                                 num_classes):
        new_median_shape = np.round(original_spacing / current_spacing *
                                    original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        input_patch_size = 1 / np.array(current_spacing)

        input_patch_size /= input_patch_size.mean()

        input_patch_size *= 1 / min(input_patch_size) * 512
        input_patch_size = np.round(input_patch_size).astype(int)
        input_patch_size = [
            min(i, j) for i, j in zip(input_patch_size, new_median_shape)
        ]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(input_patch_size,
                                                                        self.unet_featuremap_min_edge_length,
                                                                        self.unet_max_numpool,
                                                                        current_spacing)

        ref = use_this_for_batch_size_computation_3D
        here = compute_approx_vram_consumption(
            new_shp,
            network_num_pool_per_axis,
            self.unet_base_num_features,
            self.unet_max_num_filters,
            num_modalities,
            num_classes,
            pool_op_kernel_sizes,
            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[
                axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props_poolLateV2(tmp,
                                                   self.unet_featuremap_min_edge_length,
                                                   self.unet_max_numpool,
                                                   current_spacing)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[
                axis_to_be_reduced]

            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props_poolLateV2(new_shp,
                                                                            self.unet_featuremap_min_edge_length,
                                                                            self.unet_max_numpool,
                                                                            current_spacing)

            here = compute_approx_vram_consumption(
                new_shp,
                network_num_pool_per_axis,
                self.unet_base_num_features,
                self.unet_max_num_filters,
                num_modalities,
                num_classes,
                pool_op_kernel_sizes,
                conv_per_stage=self.conv_per_stage)
        input_patch_size = new_shp

        batch_size = DEFAULT_BATCH_SIZE_3D
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset
                                  * dataset_num_voxels / np.prod(
                                      input_patch_size,
                                      dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]
                                ) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm(
        )
        if use_nonzero_mask_for_normalization:
            print('We are using the nonzero mask for normalization.')
        else:
            print('We do not use the nonzero mask for normalization.')
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = [
            np.array(i) / target_spacing * np.array(j)
            for i, j in zip(spacings, sizes)
        ]

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [
            np.argwhere(np.array(self.transpose_forward) == i)[0][0]
            for i in range(3)
        ]

        median_shape = np.median(np.vstack(new_shapes), 0)
        print("The median shape of the dataset is {}.".format(median_shape))

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("The max shape in the dataset is {}.".format(max_shape))
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("The min shape in the dataset is {}.".format(min_shape))

        print("Feature maps in model will greater than {} in the bottleneck.".
              format(self.unet_featuremap_min_edge_length))

        self.plans_per_stage = list()

        target_spacing_transposed = np.array(target_spacing)[
            self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("The transposed median shape of the dataset is {}.".format(
            median_shape_transposed))

        print("Generating configuration for 3d_fullres")
        self.plans_per_stage.append(
            self.get_properties_for_stage(
                target_spacing_transposed, target_spacing_transposed,
                median_shape_transposed,
                len(self.list_of_cropped_npz_files), num_modalities,
                len(all_classes) + 1))

        architecture_input_voxels_here = np.prod(
            self.plans_per_stage[-1]['patch_size'], dtype=np.int64)
        if np.prod(
                median_shape
        ) / architecture_input_voxels_here < self.how_much_of_a_patient_must_the_network_see_at_stage0:
            gen_lowres_config = False
        else:
            gen_lowres_config = True

        if gen_lowres_config:
            print("Generating configuration for 3d_lowres")
            lowres_stage_spacing = deepcopy(target_spacing)
            num_voxels = np.prod(median_shape, dtype=np.float64)
            while num_voxels > self.how_much_of_a_patient_must_the_network_see_at_stage0 * architecture_input_voxels_here:
                max_spacing = max(lowres_stage_spacing)
                if np.any((max_spacing / lowres_stage_spacing) > 2):
                    lowres_stage_spacing[(max_spacing / lowres_stage_spacing) >
                                         2] *= 1.01
                else:
                    lowres_stage_spacing *= 1.01
                num_voxels = np.prod(
                    target_spacing / lowres_stage_spacing * median_shape,
                    dtype=np.float64)

                lowres_stage_spacing_transposed = np.array(
                    lowres_stage_spacing)[self.transpose_forward]
                lowres_props = self.get_properties_for_stage(
                    lowres_stage_spacing_transposed, target_spacing_transposed,
                    median_shape_transposed,
                    len(self.list_of_cropped_npz_files), num_modalities,
                    len(all_classes) + 1)
                architecture_input_voxels_here = np.prod(
                    lowres_props['patch_size'], dtype=np.int64)
            if 2 * np.prod(
                    lowres_props['median_patient_size_in_voxels'],
                    dtype=np.int64) < np.prod(
                        self.plans_per_stage[0][
                            'median_patient_size_in_voxels'],
                        dtype=np.int64):
                self.plans_per_stage.append(lowres_props)

        self.plans_per_stage = self.plans_per_stage[::-1]
        self.plans_per_stage = {
            i: self.plans_per_stage[i]
            for i in range(len(self.plans_per_stage))
        }

        print(self.plans_per_stage)
        print("transpose forward", self.transpose_forward)
        print("transpose backward", self.transpose_backward)

        normalization_schemes = self.determine_normalization_scheme()
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        plans = {
            'num_stages': len(list(self.plans_per_stage.keys())),
            'num_modalities': num_modalities,
            'modalities': modalities,
            'normalization_schemes': normalization_schemes,
            'dataset_properties': self.dataset_properties,
            'list_of_npz_files': self.list_of_cropped_npz_files,
            'original_spacings': spacings,
            'original_sizes': sizes,
            'preprocessed_data_folder': self.preprocessed_output_folder,
            'num_classes': len(all_classes),
            'all_classes': all_classes,
            'base_num_features': self.unet_base_num_features,
            'use_mask_for_norm': use_nonzero_mask_for_normalization,
            'keep_only_largest_region': only_keep_largest_connected_component,
            'min_region_size_per_class': min_region_size_per_class,
            'min_size_per_class': min_size_per_class,
            'transpose_forward': self.transpose_forward,
            'transpose_backward': self.transpose_backward,
            'data_identifier': self.data_identifier,
            'plans_per_stage': self.plans_per_stage,
            'preprocessor_name': self.preprocessor_name.__name__,
            'conv_per_stage': self.conv_per_stage,
        }

        self.plans = plans
        self.save_my_plans()

    def determine_normalization_scheme(self):
        schemes = OrderedDict()
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        for i in range(num_modalities):
            if modalities[i] == "CT" or modalities[i] == 'ct':
                schemes[i] = "CT"
            elif modalities[i] == 'noNorm':
                schemes[i] = "noNorm"
            else:
                schemes[i] = "nonCT"
        return schemes

    def save_properties_of_cropped(self, case_identifier, properties):
        with open(
                join_paths(self.folder_with_cropped_data,
                           "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def load_properties_of_cropped(self, case_identifier):
        with open(
                join_paths(self.folder_with_cropped_data,
                           "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def determine_whether_to_use_mask_for_norm(self):
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))
        use_nonzero_mask_for_norm = OrderedDict()

        for i in range(num_modalities):
            if "CT" in modalities[i]:
                use_nonzero_mask_for_norm[i] = False
            else:
                all_size_reductions = []
                for k in self.dataset_properties['size_reductions'].keys():
                    all_size_reductions.append(self.dataset_properties[
                        'size_reductions'][k])

                if np.median(all_size_reductions) < 3 / 4.:
                    print("using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = True
                else:
                    print("not using nonzero mask for normalization")
                    use_nonzero_mask_for_norm[i] = False

        for c in self.list_of_cropped_npz_files:
            case_identifier = get_case_identifier_from_npz(c)
            properties = self.load_properties_of_cropped(case_identifier)
            properties['use_nonzero_mask_for_norm'] = use_nonzero_mask_for_norm
            self.save_properties_of_cropped(case_identifier, properties)
        use_nonzero_mask_for_normalization = use_nonzero_mask_for_norm
        return use_nonzero_mask_for_normalization

    def write_normalization_scheme_to_patients(self):
        for c in self.list_of_cropped_npz_files:
            case_identifier = get_case_identifier_from_npz(c)
            properties = self.load_properties_of_cropped(case_identifier)
            properties['use_nonzero_mask_for_norm'] = self.plans[
                'use_mask_for_norm']
            self.save_properties_of_cropped(case_identifier, properties)

    def run_preprocessing(self, num_threads):
        if os.path.isdir(
                join_paths(self.preprocessed_output_folder,
                           "gt_segmentations")):
            shutil.rmtree(
                join_paths(self.preprocessed_output_folder, "gt_segmentations"))
        shutil.copytree(
            join_paths(self.folder_with_cropped_data, "gt_segmentations"),
            join_paths(self.preprocessed_output_folder, "gt_segmentations"))
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties'][
            'intensityproperties']
        preprocessor_class = self.preprocessor_name
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(
            normalization_schemes, use_nonzero_mask_for_normalization,
            self.transpose_forward, intensityproperties)
        target_spacings = [
            i["current_spacing"] for i in self.plans_per_stage.values()
        ]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads,
                                                           (list, tuple)):
            num_threads = (8, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads,
                                                          (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run(target_spacings, self.folder_with_cropped_data,
                         self.preprocessed_output_folder,
                         self.plans['data_identifier'], num_threads)


class ExperimentPlanner2D_v21(ExperimentPlanner):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner2D_v21, self).__init__(
            folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1_2D"
        self.plans_fname = join_paths(self.preprocessed_output_folder,
                                      "nnUNetPlansv2.1_plans_2D.pkl")
        self.unet_base_num_features = 32
        self.unet_max_num_filters = 512
        self.unet_max_numpool = 999

        self.preprocessor_name = PreprocessorFor2D

    def get_properties_for_stage(self, current_spacing, original_spacing,
                                 original_shape, num_cases, num_modalities,
                                 num_classes):

        new_median_shape = np.round(original_spacing / current_spacing *
                                    original_shape).astype(int)

        dataset_num_voxels = np.prod(
            new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shape, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        ref = use_this_for_batch_size_computation_2D * DEFAULT_BATCH_SIZE_2D / 2
        here = compute_approx_vram_consumption(
            new_shape,
            network_num_pool_per_axis,
            30,
            self.unet_max_num_filters,
            num_modalities,
            num_classes,
            pool_op_kernel_sizes,
            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shape /
                                            new_median_shape[1:])[-1]

            tmp = deepcopy(new_shape)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[
                axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = get_pool_and_conv_props(
                current_spacing[1:], tmp, self.unet_featuremap_min_edge_length,
                self.unet_max_numpool)
            new_shape[axis_to_be_reduced] -= shape_must_be_divisible_by_new[
                axis_to_be_reduced]

            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shape, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], new_shape,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool)

            here = compute_approx_vram_consumption(
                new_shape,
                network_num_pool_per_axis,
                self.unet_base_num_features,
                self.unet_max_num_filters,
                num_modalities,
                num_classes,
                pool_op_kernel_sizes,
                conv_per_stage=self.conv_per_stage)

        batch_size = int(np.floor(ref / here) * 2)
        input_patch_size = new_shape

        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset
                                  * dataset_num_voxels / np.prod(
                                      input_patch_size,
                                      dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan

    def plan_experiment(self):
        use_nonzero_mask_for_normalization = self.determine_whether_to_use_mask_for_norm(
        )
        if use_nonzero_mask_for_normalization:
            print('We are using the nonzero mask for normalization.')
        else:
            print('We are not using the nonzero mask for normalization.')

        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']
        all_classes = self.dataset_properties['all_classes']
        modalities = self.dataset_properties['modalities']
        num_modalities = len(list(modalities.keys()))

        target_spacing = self.get_target_spacing()
        new_shapes = np.array([
            np.array(i) / target_spacing * np.array(j)
            for i, j in zip(spacings, sizes)
        ])

        max_spacing_axis = np.argmax(target_spacing)
        remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
        self.transpose_forward = [max_spacing_axis] + remaining_axes
        self.transpose_backward = [
            np.argwhere(np.array(self.transpose_forward) == i)[0][0]
            for i in range(3)
        ]

        median_shape = np.median(np.vstack(new_shapes), 0)
        print("The median shape of the dataset is {}.".format(median_shape))

        max_shape = np.max(np.vstack(new_shapes), 0)
        print("The max shape in the dataset is {}.".format(max_shape))
        min_shape = np.min(np.vstack(new_shapes), 0)
        print("The min shape in the dataset is {}.".format(min_shape))

        print("Feature maps in model will greater than {} in the bottleneck.".
              format(self.unet_featuremap_min_edge_length))

        target_spacing_transposed = np.array(target_spacing)[
            self.transpose_forward]
        median_shape_transposed = np.array(median_shape)[self.transpose_forward]
        print("The transposed median shape of the dataset is {}.".format(
            median_shape_transposed))

        self.plans_per_stage = {
            0: self.get_properties_for_stage(
                target_spacing_transposed,
                target_spacing_transposed,
                median_shape_transposed,
                num_cases=len(self.list_of_cropped_npz_files),
                num_modalities=num_modalities,
                num_classes=len(all_classes) + 1)
        }

        normalization_schemes = self.determine_normalization_scheme()
        only_keep_largest_connected_component, min_size_per_class, min_region_size_per_class = None, None, None

        plans = {
            'num_stages': len(list(self.plans_per_stage.keys())),
            'num_modalities': num_modalities,
            'modalities': modalities,
            'normalization_schemes': normalization_schemes,
            'dataset_properties': self.dataset_properties,
            'list_of_npz_files': self.list_of_cropped_npz_files,
            'original_spacings': spacings,
            'original_sizes': sizes,
            'preprocessed_data_folder': self.preprocessed_output_folder,
            'num_classes': len(all_classes),
            'all_classes': all_classes,
            'base_num_features': self.unet_base_num_features,
            'use_mask_for_norm': use_nonzero_mask_for_normalization,
            'keep_only_largest_region': only_keep_largest_connected_component,
            'min_region_size_per_class': min_region_size_per_class,
            'min_size_per_class': min_size_per_class,
            'transpose_forward': self.transpose_forward,
            'transpose_backward': self.transpose_backward,
            'data_identifier': self.data_identifier,
            'plans_per_stage': self.plans_per_stage,
            'preprocessor_name': self.preprocessor_name.__name__,
        }

        self.plans = plans
        self.save_my_plans()


class ExperimentPlanner3D_v21(ExperimentPlanner):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlanner3D_v21, self).__init__(
            folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "nnUNetData_plans_v2.1"
        self.plans_fname = join_paths(self.preprocessed_output_folder,
                                      "nnUNetPlansv2.1_plans_3D.pkl")
        self.unet_base_num_features = 32

    def get_target_spacing(self):
        spacings = self.dataset_properties['all_spacings']
        sizes = self.dataset_properties['all_sizes']

        target = np.percentile(
            np.vstack(spacings), self.target_spacing_percentile, 0)
        target_size = np.percentile(
            np.vstack(sizes), self.target_spacing_percentile, 0)
        target_size_mm = np.array(target) * np.array(target_size)

        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (
            self.anisotropy_threshold * max(other_spacings))
        has_aniso_voxels = target_size[
            worst_spacing_axis] * self.anisotropy_threshold < min(other_sizes)

        if has_aniso_spacing and has_aniso_voxels:
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis,
                                                        10)
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(
                    max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target

    def get_properties_for_stage(self, current_spacing, original_spacing,
                                 original_shape, num_cases, num_modalities,
                                 num_classes):
        new_median_shape = np.round(original_spacing / current_spacing *
                                    original_shape).astype(int)
        dataset_num_voxels = np.prod(new_median_shape) * num_cases

        input_patch_size = 1 / np.array(current_spacing)
        input_patch_size /= input_patch_size.mean()

        input_patch_size *= 1 / min(input_patch_size) * 512
        input_patch_size = np.round(input_patch_size).astype(int)

        input_patch_size = [
            min(i, j) for i, j in zip(input_patch_size, new_median_shape)
        ]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, shape_must_be_divisible_by = get_pool_and_conv_props(
            current_spacing, input_patch_size,
            self.unet_featuremap_min_edge_length, self.unet_max_numpool)

        ref = use_this_for_batch_size_computation_3D * self.unet_base_num_features / BASE_NUM_FEATURES_3D
        here = compute_approx_vram_consumption(
            new_shp,
            network_num_pool_per_axis,
            self.unet_base_num_features,
            self.unet_max_num_filters,
            num_modalities,
            num_classes,
            pool_op_kernel_sizes,
            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape)[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[
                axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing, tmp,
                                        self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool,
                                        )
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[
                axis_to_be_reduced]

            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, shape_must_be_divisible_by = get_pool_and_conv_props(
                current_spacing,
                new_shp,
                self.unet_featuremap_min_edge_length,
                self.unet_max_numpool, )

            here = compute_approx_vram_consumption(
                new_shp,
                network_num_pool_per_axis,
                self.unet_base_num_features,
                self.unet_max_num_filters,
                num_modalities,
                num_classes,
                pool_op_kernel_sizes,
                conv_per_stage=self.conv_per_stage)
        input_patch_size = new_shp

        batch_size = DEFAULT_BATCH_SIZE_3D  # This is what wirks with 128**3
        batch_size = int(np.floor(max(ref / here, 1) * batch_size))

        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset
                                  * dataset_num_voxels / np.prod(
                                      input_patch_size,
                                      dtype=np.int64)).astype(int)
        max_batch_size = max(max_batch_size, self.unet_min_batch_size)
        batch_size = max(1, min(batch_size, max_batch_size))

        do_dummy_2D_data_aug = (max(input_patch_size) / input_patch_size[0]
                                ) > self.anisotropy_threshold

        plan = {
            'batch_size': batch_size,
            'num_pool_per_axis': network_num_pool_per_axis,
            'patch_size': input_patch_size,
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': do_dummy_2D_data_aug,
            'pool_op_kernel_sizes': pool_op_kernel_sizes,
            'conv_kernel_sizes': conv_kernel_sizes,
        }
        return plan
