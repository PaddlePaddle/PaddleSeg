# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import os.path as osp
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

from ..base.utils.cache import persist


@persist(0)
def check_dataset(dataset_dir, dataset_type):
    if dataset_type == 'Dataset':
        # Custom dataset
        dataset_dir = osp.abspath(dataset_dir)
        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            return False

        tags = ['train', 'val']
        sample_cnts = dict()
        sample_paths = defaultdict(list)
        im_sizes = defaultdict(Counter)
        num_classes = 0
        for tag in tags:
            file_list = osp.join(dataset_dir, f'{tag}.txt')
            if not osp.exists(file_list):
                # All file lists must exist
                return False
            else:
                with open(file_list, 'r') as f:
                    all_lines = f.readlines()

                # Each line corresponds to a sample
                sample_cnts[tag] = len(all_lines)

                for line in all_lines:
                    # Use a default delimiter
                    parts = line.rstrip().split()
                    if len(parts) != 2:
                        # Each line must contain two parts
                        return False

                    img_path, lab_path = parts
                    # NOTE: According to PaddleSeg rules, paths recorded in file list are
                    # the paths relative to dataset root dir.
                    # We populate the list with a dict
                    sample_paths[tag].append({'img': img_path, 'lab': lab_path})

                    img_path = osp.join(dataset_dir, img_path)
                    lab_path = osp.join(dataset_dir, lab_path)
                    if not osp.exists(img_path) or not osp.exists(lab_path):
                        # Every path has to be valid
                        return False

                    img = Image.open(img_path)
                    lab = Image.open(lab_path)
                    im_shape = img.size
                    lab_shape = lab.size
                    if img.mode != 'RGB':
                        # We require RGB images
                        return False
                    if lab.mode not in ('L', 'P'):
                        # We require gray-scale masks or pseudo-color masks
                        return False
                    if im_shape != lab_shape:
                        # Image and mask must have the same shape
                        return False

                    im_sizes[tag][tuple(im_shape)] += 1
                    # We have to load mask to memory
                    lab = np.asarray(lab)

                    num_classes = max(num_classes, int(lab.max() + 1))

        meta = dict()

        meta['num_classes'] = num_classes

        meta['train.samples'] = sample_cnts['train']
        meta['train.im_sizes'] = im_sizes['train']
        meta['train.sample_paths'] = sample_paths['train']

        meta['val.samples'] = sample_cnts['val']
        meta['val.im_sizes'] = im_sizes['val']
        meta['val.sample_paths'] = sample_paths['val']

        # PaddleSeg does not use test subset
        meta['test.samples'] = None
        meta['test.im_sizes'] = None
        meta['test.sample_paths'] = None

        return meta
    else:
        raise ValueError(f"{dataset_type} is not supported.")
