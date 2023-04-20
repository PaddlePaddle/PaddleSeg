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
import glob
import json
import numpy as np

from multiprocessing import Pool
from collections import OrderedDict
from sklearn.model_selection import KFold
from typing import List

from medicalseg.datasets import MedicalDataset
from medicalseg.cvlibs import manager
from nnunet.transforms import get_moreDA_augmentation, default_2D_augmentation_params, default_3D_augmentation_params
from nnunet.transforms import rotate_coords_3d, rotate_coords_2d, NumpyToTensor

from .dataloader import DataLoader2D, DataLoader3D
from tools.preprocess_utils import verify_dataset_integrity, convert_to_decathlon, crop, ExperimentPlanner2D_v21, ExperimentPlanner3D_v21, DatasetAnalyzer


@manager.DATASETS.add_component
class MSDDataset(MedicalDataset):
    def __init__(self,
                 plans_name=None,
                 num_batches_per_epoch=250,
                 fold=0,
                 dataset_directory=None,
                 raw_data_dir=None,
                 decathlon_dir=None,
                 cropped_data_dir=None,
                 preprocessed_dir=None,
                 plan2d=False,
                 plan3d=False,
                 stage=None,
                 cascade=False,
                 dataset_root=None,
                 result_dir=None,
                 use_multi_augmenter=True,
                 transforms=None,
                 num_classes=None,
                 unpack_data=True,
                 mode='train',
                 ignore_index=255,
                 num_threads=4):
        self.raw_data_dir = raw_data_dir
        self.decathlon_dir = decathlon_dir
        self.cropped_data_dir = cropped_data_dir
        self.num_threads = num_threads
        self.plan2d = plan2d
        self.plan3d = plan3d
        self.preprocessed_dir = preprocessed_dir
        self.preprocess_and_generate_plans()
        self.cascade = cascade
        self.use_multi_augmenter = use_multi_augmenter

        self.folder_with_segs_from_prev_stage = os.path.join(preprocessed_dir,
                                                             'pred_next_stage')

        self.dataset_directory = preprocessed_dir
        self.unpack_data = unpack_data
        self.stage = stage

        self.fold = fold
        self.num_batches_per_epoch = num_batches_per_epoch
        self.plans_path = os.path.join(self.preprocessed_dir, plans_name)
        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.file_list = list()
        self.mode = mode.lower()
        self.ignore_index = ignore_index

        self.oversample_foreground_percent = 0.33
        self.pin_memory = True
        self.initialize()

        self.convert_to_tensor = NumpyToTensor(['data', 'target'], 'float32')

    def __len__(self):
        return self.num_batches_per_epoch

    def __getitem__(self, idx):
        data_dict = next(self.gen)
        data_dict = self.convert_to_tensor(**data_dict)
        return data_dict['data'], data_dict['target']

    def preprocess_and_generate_plans(self):
        if not os.path.exists(self.decathlon_dir):
            print("{} not found, convert data to decathlon.".format(
                self.decathlon_dir))
            convert_to_decathlon(
                input_folder=self.raw_data_dir,
                output_folder=self.decathlon_dir,
                num_processes=self.num_threads)
            verify_dataset_integrity(
                self.decathlon_dir, default_num_threads=self.num_threads)
        else:
            print(
                "Found existed {}, please ensure your dataset is preprocessed correctly!!!".
                format(self.decathlon_dir))

        if not os.path.exists(self.cropped_data_dir):
            print("{} not found, crop data, it may be time consuming.".format(
                self.cropped_data_dir))
            crop(
                self.decathlon_dir,
                self.cropped_data_dir,
                override=False,
                num_threads=self.num_threads)
        else:
            print(
                "Found existed {}, please ensure your dataset is preprocessed correctly!!!".
                format(self.cropped_data_dir))

        if not os.path.exists(
                os.path.join(self.cropped_data_dir, 'dataset_properties.pkl')):
            print("{} not exist, analyzing dataset...".format(
                os.path.join(self.cropped_data_dir, 'dataset_properties.pkl')))
            with open(
                    os.path.join(
                        os.path.join(self.cropped_data_dir, 'dataset.json')),
                    'rb') as f:
                dataset_json = json.load(f)
            modalities = list(dataset_json["modality"].values())
            collect_intensityproperties = True if (
                ("CT" in modalities) or ("ct" in modalities)) else False
            dataset_analyzer = DatasetAnalyzer(
                self.cropped_data_dir,
                overwrite=True,
                num_processes=self.num_threads)
            dataset_analyzer.analyze_dataset(collect_intensityproperties)
        else:
            print(
                "Found existed dataset_properties.pkl, please ensure your dataset_properties is preprocessed correctly!!!"
            )

        if self.plan2d:
            exp_planner_2d = ExperimentPlanner2D_v21(self.cropped_data_dir,
                                                     self.preprocessed_dir)
            if not os.path.exists(exp_planner_2d.plans_fname):
                print("{} not exists, generating plans, please wait a minute.".
                      format(exp_planner_2d.plans_fname))
                exp_planner_2d.plan_experiment()
                exp_planner_2d.run_preprocessing(self.num_threads)
            else:
                print(
                    "Found existed plan file, please ensure your plan file is preprocessed correctly!!!"
                )

        if self.plan3d:
            exp_planner_3d = ExperimentPlanner3D_v21(self.cropped_data_dir,
                                                     self.preprocessed_dir)
            if not os.path.exists(exp_planner_3d.plans_fname):
                print(
                    "{} already exists, generating plans, please wait a minute.".
                    format(exp_planner_3d.plans_fname))
                exp_planner_3d.plan_experiment()
                exp_planner_3d.run_preprocessing(self.num_threads)
            else:
                print(
                    "Found existed plan file, please ensure your plan file is preprocessed correctly!!!"
                )

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            if self.stage == 0:
                dl_tr = DataLoader3D(
                    self.dataset_tr,
                    self.basic_generator_patch_size,
                    self.patch_size,
                    self.batch_size,
                    False,
                    oversample_foreground_percent=self.
                    oversample_foreground_percent,
                    pad_mode="constant",
                    pad_sides=self.pad_all_sides,
                    memmap_mode='r')
                dl_val = DataLoader3D(
                    self.dataset_val,
                    self.patch_size,
                    self.patch_size,
                    self.batch_size,
                    False,
                    oversample_foreground_percent=self.
                    oversample_foreground_percent,
                    pad_mode="constant",
                    pad_sides=self.pad_all_sides,
                    memmap_mode='r')
            else:
                dl_tr = DataLoader3D(
                    self.dataset_tr,
                    self.basic_generator_patch_size,
                    self.patch_size,
                    self.batch_size,
                    has_prev_stage=self.cascade,
                    oversample_foreground_percent=self.
                    oversample_foreground_percent,
                    pad_mode="constant",
                    pad_sides=self.pad_all_sides)
                dl_val = DataLoader3D(
                    self.dataset_val,
                    self.patch_size,
                    self.patch_size,
                    self.batch_size,
                    has_prev_stage=self.cascade,
                    oversample_foreground_percent=self.
                    oversample_foreground_percent,
                    pad_mode="constant",
                    pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.
                oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode='r')
            dl_val = DataLoader2D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                oversample_foreground_percent=self.
                oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode='r')
        return dl_tr, dl_val

    def initialize(self):
        self.load_plans_file()
        self.process_plans(self.plans)
        self.setup_DA_params()
        self.folder_with_preprocessed_data = os.path.join(
            self.dataset_directory,
            self.plans['data_identifier'] + "_stage%d" % self.stage)
        if len(
                glob.glob(
                    os.path.join(self.folder_with_preprocessed_data,
                                 '*.npy'))) <= 0:
            self.unpack_data = False
            print("unpacking dataset")
            unpack_dataset(
                self.folder_with_preprocessed_data, threads=self.num_threads)
            print("done")
        else:
            print("found unpacked dataset.")

        dl_tr, dl_val = self.get_basic_generators()
        tr_gen, val_gen = get_moreDA_augmentation(
            dl_tr,
            dl_val,
            self.data_aug_params['patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory,
            use_multi_augmenter=self.use_multi_augmenter)
        if self.mode == 'train':
            self.gen = tr_gen
        else:
            self.gen = val_gen

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1 / np.cumprod(
                np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi,
                                                  30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi,
                                                  30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi,
                                                  30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                print("Using dummy2d data augmentation")
                self.data_aug_params[
                    "elastic_deform_alpha"] = default_2D_augmentation_params[
                        "elastic_deform_alpha"]
                self.data_aug_params[
                    "elastic_deform_sigma"] = default_2D_augmentation_params[
                        "elastic_deform_sigma"]
                self.data_aug_params[
                    "rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (
                    -15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params[
            "mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:], self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[
                0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size, self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params[
            'patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

        if self.stage == 1 and self.cascade:
            self.data_aug_params["num_cached_per_thread"] = 2

            self.data_aug_params['move_last_seg_channel_to_data'] = True
            self.data_aug_params['cascade_do_cascade_augmentations'] = True

            self.data_aug_params['cascade_random_binary_transform_p'] = 0.4
            self.data_aug_params[
                'cascade_random_binary_transform_p_per_label'] = 1
            self.data_aug_params['cascade_random_binary_transform_size'] = (1,
                                                                            8)

            self.data_aug_params['cascade_remove_conn_comp_p'] = 0.2
            self.data_aug_params[
                'cascade_remove_conn_comp_max_size_percent_threshold'] = 0.15
            self.data_aug_params[
                'cascade_remove_conn_comp_fill_with_other_class_p'] = 0.0

            self.data_aug_params['selected_seg_channels'] = [0, 1]
            self.data_aug_params['all_segmentation_labels'] = list(
                range(1, self.num_classes))

    def load_plans_file(self):
        with open(self.plans_path, 'rb') as f:
            self.plans = pickle.load(f)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it..."
            )
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans[
                'pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it..."
            )
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)
                                          ] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None
        self.intensity_properties = plans['dataset_properties'][
            'intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1

        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans[
            'keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None

        if plans.get('transpose_forward') is None or plans.get(
                'transpose_backward') is None:
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" %
                               str(self.patch_size))

        if "conv_per_stage" in plans.keys():
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def do_split(self):
        splits_file = os.path.join(self.dataset_directory, "splits_final.pkl")
        if not os.path.isfile(splits_file):
            print("Creating new split...")
            splits = []
            all_keys_sorted = np.sort(list(self.dataset.keys()))
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx,
                    test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
            with open(splits_file, 'wb') as f:
                pickle.dump(splits, f)

        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        # cascade stage 2
        if self.stage == 1 and self.cascade:
            for k in self.dataset:
                self.dataset[k]['seg_from_prev_stage_file'] = os.path.join(
                    self.folder_with_segs_from_prev_stage,
                    k + "_segFromPrevStage.npz")

        print("dataset split over! dataset mode: {}, keys: {}".format(
            self.mode, tr_keys if self.mode == 'train' else val_keys))


def unpack_dataset(folder, threads=8, key="data"):
    p = Pool(threads)
    npz_files = [
        os.path.join(folder, path) for path in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, path)) and path.endswith('.npz')
    ]
    npz_files.sort()
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def subfiles(folder: str,
             join: bool=True,
             prefix: str=None,
             suffix: str=None,
             sort: bool=True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i) for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i)) and
        (prefix is None or
         i.startswith(prefix)) and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)

    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)),
                       final_shape)), 0)
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)),
                       final_shape)), 0)
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)),
                       final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)),
            0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not os.path.isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def get_case_identifiers(folder):
    case_identifiers = [
        i[:-4] for i in os.listdir(folder)
        if i.endswith("npz") and (i.find("segFromPrevStage") == -1)
    ]
    return case_identifiers


def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    print('loading dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = os.path.join(folder, "%s.npz" % c)

        dataset[c]['properties_file'] = os.path.join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = os.path.join(
                folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            with open(dataset[i]['properties_file'], 'rb') as f:
                dataset[i]['properties'] = pickle.load(f)
    return dataset
