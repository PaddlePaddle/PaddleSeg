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

from collections import OrderedDict

# TODO: We need a lightweight RDBMS to handle these tables.

model_zoo = OrderedDict()
repo_zoo = OrderedDict()

MODEL_INFO_REQUIRED_KEYS = ('model_name', 'repo', 'config_path', 'type',
                            'auto_compression_config_path')
MODEL_INFO_PRIMARY_KEY = 'model_name'
assert MODEL_INFO_PRIMARY_KEY in MODEL_INFO_REQUIRED_KEYS
REPO_INFO_REQUIRED_KEYS = ('repo_name', 'repo_cls', 'root_path')
REPO_INFO_PRIMARY_KEY = 'repo_name'
assert REPO_INFO_PRIMARY_KEY in REPO_INFO_REQUIRED_KEYS

# Relations:
# 'repo' in model info <-> 'repo_name' in repo info


def _validate_model_info(model_info):
    for key in MODEL_INFO_REQUIRED_KEYS:
        if key not in model_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def _validate_repo_info(repo_info):
    for key in REPO_INFO_REQUIRED_KEYS:
        if key not in repo_info:
            raise KeyError(f"Key '{key}' is required, but not found.")


def register_model_info(data):
    global model_zoo
    _validate_model_info(data)
    prim_key = data[MODEL_INFO_PRIMARY_KEY]
    model_zoo[prim_key] = data


def register_repo_info(data):
    global repo_zoo
    _validate_repo_info(data)
    prim_key = data[REPO_INFO_PRIMARY_KEY]
    repo_zoo[prim_key] = data


def get_registered_model_info(prim_key):
    return model_zoo[prim_key]


def get_registered_repo_info(prim_key):
    return repo_zoo[prim_key]


def build_repo_from_model_info(model_info):
    repo_name = model_info['repo']
    # `repo_name` being the primary key of repo info
    repo_info = repo_zoo[repo_name]
    repo_cls = repo_info['repo_cls']
    repo_root_path = repo_info['root_path']
    return repo_cls(root_path=repo_root_path)
