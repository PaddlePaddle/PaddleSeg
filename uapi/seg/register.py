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

from ..base.register import register_model_info, register_suite_info
from .model import SegModel
from .runner import SegRunner

# XXX: Hard-code relative path of repo root dir
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
register_suite_info({
    'suite_name': 'Seg',
    'model': SegModel,
    'runner': SegRunner,
    'runner_root_path': REPO_ROOT_PATH,
    'supported_api_list':
    ['train', 'predict', 'export', 'infer', 'compression']
})

PPHUMANSEG_LITE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs',
                                    'pp_humanseg_lite',
                                    'pp_humanseg_lite_mini_supervisely.yml')
register_model_info({
    'model_name': 'pphumanseg_lite',
    'suite': 'Seg',
    'config_path': PPHUMANSEG_LITE_CFG_PATH,
    'auto_compression_config_path': PPHUMANSEG_LITE_CFG_PATH
})
