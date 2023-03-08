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
import json
import os
import pickle
import numpy as np
from skimage.morphology import label
from collections import OrderedDict
from multiprocessing import Pool

from .path_utils import join_paths


class DatasetAnalyzer:
    def __init__(self,
                 folder_with_cropped_data,
                 overwrite=True,
                 dataset_property_pkl="dataset_properties.pkl",
                 data_json="dataset.json",
                 intensityproperties_file="intensityproperties.pkl",
                 num_processes=8):
        self.overwrite = overwrite
        self.num_processes = num_processes
        self.folder_with_cropped_data = folder_with_cropped_data
        self.dataset_property_pkl = join_paths(folder_with_cropped_data,
                                               dataset_property_pkl)
        self.data_json = join_paths(self.folder_with_cropped_data, data_json)
        assert os.path.isfile(
            self.data_json
        ), "{} needs to be in {} but not found. Please ensure your folder including cropped data.".format(
            data_json, folder_with_cropped_data)
        self.intensityproperties_file = join_paths(
            self.folder_with_cropped_data, intensityproperties_file)

        self.sizes = self.spacings = None
        self.patient_identifiers = self.get_patient_identifiers_from_cropped_files(
            self.folder_with_cropped_data)

    def get_patient_identifiers_from_cropped_files(self, folder):
        file_list = []
        for file_name in os.listdir(folder):
            if os.path.isfile(join_paths(
                    folder, file_name)) and file_name.endswith(".npz"):
                file_list.append(join_paths(folder, file_name))
        return [i.split("/")[-1].split('.')[0] for i in file_list]

    def load_properties_of_cropped(self, case_identifier):
        with open(
                join_paths(self.folder_with_cropped_data,
                           "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def get_classes(self):
        with open(self.data_json, 'rb') as f:
            dataset_json = json.load(f)
        return dataset_json['labels']

    def get_sizes_and_spacings_after_cropping(self):
        sizes = []
        spacings = []
        for c in self.patient_identifiers:
            properties = self.load_properties_of_cropped(c)
            sizes.append(properties["size_after_cropping"])
            spacings.append(properties["original_spacing"])
        return sizes, spacings

    def get_modalities(self):
        with open(self.data_json, 'rb') as f:
            dataset_json = json.load(f)
        modalities = dataset_json["modality"]
        modalities = {int(k): modalities[k] for k in modalities.keys()}
        return modalities

    def get_size_reduction_by_cropping(self):
        size_reduction = OrderedDict()
        for p in self.patient_identifiers:
            props = self.load_properties_of_cropped(p)
            shape_before_crop = props["original_size_of_raw_data"]
            shape_after_crop = props['size_after_cropping']
            size_red = np.prod(shape_after_crop) / np.prod(shape_before_crop)
            size_reduction[p] = size_red
        return size_reduction

    def _get_voxels_in_foreground(self, patient_identifier, modality_id):
        all_data = np.load(
            join_paths(self.folder_with_cropped_data, patient_identifier) +
            ".npz")['data']
        modality = all_data[modality_id]
        mask = all_data[-1] > 0
        voxels = list(modality[mask][::10])
        return voxels

    def _compute_stats(self, voxels):
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)
        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

    def collect_intensity_properties(self, num_modalities):
        if self.overwrite or not os.path.isfile(self.intensityproperties_file):
            p = Pool(self.num_processes)

            results = OrderedDict()
            for mod_id in range(num_modalities):
                results[mod_id] = OrderedDict()
                voxels = p.starmap(self._get_voxels_in_foreground,
                                   zip(self.patient_identifiers, [mod_id] *
                                       len(self.patient_identifiers)))
                all_voxels = []
                for voxel in voxels:
                    all_voxels += voxel

                median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(
                    all_voxels)
                local_props = p.map(self._compute_stats, voxels)
                props_per_case = OrderedDict()
                for i, pat in enumerate(self.patient_identifiers):
                    props_per_case[pat] = OrderedDict()
                    props_per_case[pat]['median'] = local_props[i][0]
                    props_per_case[pat]['mean'] = local_props[i][1]
                    props_per_case[pat]['sd'] = local_props[i][2]
                    props_per_case[pat]['mn'] = local_props[i][3]
                    props_per_case[pat]['mx'] = local_props[i][4]
                    props_per_case[pat]['percentile_99_5'] = local_props[i][5]
                    props_per_case[pat]['percentile_00_5'] = local_props[i][6]

                results[mod_id]['local_props'] = props_per_case
                results[mod_id]['median'] = median
                results[mod_id]['mean'] = mean
                results[mod_id]['sd'] = sd
                results[mod_id]['mn'] = mn
                results[mod_id]['mx'] = mx
                results[mod_id]['percentile_99_5'] = percentile_99_5
                results[mod_id]['percentile_00_5'] = percentile_00_5

            p.close()
            p.join()
            with open(self.intensityproperties_file, 'wb') as f:
                pickle.dump(results, f)
        else:
            with open(self.intensityproperties_file, 'rb') as f:
                results = pickle.load(f)
        return results

    def analyze_dataset(self, collect_intensityproperties=True):
        sizes, spacings = self.get_sizes_and_spacings_after_cropping()

        classes = self.get_classes()
        all_classes = [int(i) for i in classes.keys() if int(i) > 0]

        modalities = self.get_modalities()
        if collect_intensityproperties:
            intensityproperties = self.collect_intensity_properties(
                len(modalities))
        else:
            intensityproperties = None

        size_reductions = self.get_size_reduction_by_cropping()
        dataset_properties = {}
        dataset_properties['all_sizes'] = sizes
        dataset_properties['all_spacings'] = spacings
        dataset_properties['all_classes'] = all_classes
        dataset_properties['modalities'] = modalities
        dataset_properties['intensityproperties'] = intensityproperties
        dataset_properties['size_reductions'] = size_reductions

        with open(self.dataset_property_pkl, 'wb') as f:
            pickle.dump(dataset_properties, f)
        return dataset_properties
