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
from .config import SegConfig

# XXX: Hard-code relative path of repo root dir
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
register_suite_info({
    'suite_name': 'Seg',
    'model': SegModel,
    'runner': SegRunner,
    'config': SegConfig,
    'runner_root_path': REPO_ROOT_PATH
})

# DeepLab V3+
DEEPLABV3P_R50_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'deeplabv3p',
    'deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'deeplabv3p_r50',
    'suite': 'Seg',
    'config_path': DEEPLABV3P_R50_CFG_PATH,
    'auto_compression_config_path': DEEPLABV3P_R50_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
DEEPLABV3P_R101_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'deeplabv3p',
    'deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'deeplabv3p_r101',
    'suite': 'Seg',
    'config_path': DEEPLABV3P_R101_CFG_PATH,
    'auto_compression_config_path': DEEPLABV3P_R101_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})

# MobileSeg
MOBILESEG_MV2_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'mobileseg',
    'mobileseg_mobilenetv2_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'mobileseg_mv2',
    'suite': 'Seg',
    'config_path': MOBILESEG_MV2_CFG_PATH,
    'auto_compression_config_path': MOBILESEG_MV2_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
MOBILESEG_MV3_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'mobileseg',
    'mobileseg_mobilenetv3_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'mobileseg_mv3',
    'suite': 'Seg',
    'config_path': MOBILESEG_MV3_CFG_PATH,
    'auto_compression_config_path': MOBILESEG_MV3_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})

# PP-HumanSeg
PPHUMANSEG_LITE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs',
                                    'pp_humanseg_lite',
                                    'pp_humanseg_lite_mini_supervisely.yml')
register_model_info({
    'model_name': 'pphumanseg_lite',
    'suite': 'Seg',
    'config_path': PPHUMANSEG_LITE_CFG_PATH,
    'auto_compression_config_path': PPHUMANSEG_LITE_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})

# PP-LiteSeg
PP_LITESEG_STDC1_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'pp_liteseg',
    'pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml')
register_model_info({
    'model_name': 'pp_liteseg_stdc1',
    'suite': 'Seg',
    'config_path': PP_LITESEG_STDC1_CFG_PATH,
    'auto_compression_config_path': PP_LITESEG_STDC1_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
PP_LITESEG_STDC2_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'pp_liteseg',
    'pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml')
register_model_info({
    'model_name': 'pp_liteseg_stdc2',
    'suite': 'Seg',
    'config_path': PP_LITESEG_STDC2_CFG_PATH,
    'auto_compression_config_path': PP_LITESEG_STDC2_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})

# RTFormer
RTFORMER_BASE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'rtformer',
                                  'rtformer_base_cityscapes_1024x512_120k.yml')
register_model_info({
    'model_name': 'rtformer_base',
    'suite': 'Seg',
    'config_path': RTFORMER_BASE_CFG_PATH,
    'auto_compression_config_path': RTFORMER_BASE_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})

# SegFormer
SEGFORMER_B0_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'segformer',
                                 'segformer_b0_cityscapes_1024x512_160k.yml')
register_model_info({
    'model_name': 'segformer_b0',
    'suite': 'Seg',
    'config_path': SEGFORMER_B0_CFG_PATH,
    'auto_compression_config_path': SEGFORMER_B0_CFG_PATH,
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
