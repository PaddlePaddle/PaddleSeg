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


def check_dataset(dataset_dir, dataset_type):
    if dataset_type == 'Dataset':
        # Custom dataset
        dataset_dir = osp.abspath(dataset_dir)
        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            return False
        tags = ['train', 'val']
        sample_cnts = dict()
        for tag in tags:
            file_list = osp.join(dataset_dir, f'{tag}.txt')
            if not osp.exists(file_list):
                # All file lists must exist
                return False
            else:
                with open(file_list, 'r') as f:
                    all_lines = f.readlines()
                # TODO: Check validity of each image
                # Each line corresponds to a sample
                sample_cnts[tag] = len(all_lines)
        # PaddleSeg does not use test subset
        return [sample_cnts['train'], sample_cnts['val'], None]
    else:
        raise ValueError(f"{dataset_type} is not supported.")
