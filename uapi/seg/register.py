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
_file_path = osp.realpath(__file__)
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(_file_path), '..', '..'))
register_suite_info({
    'suite_name': 'Seg',
    'model': SegModel,
    'runner': SegRunner,
    'config': SegConfig,
    'runner_root_path': REPO_ROOT_PATH
})

# BiSeNet V2
BISENETV2_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'bisenet',
                              'bisenet_cityscapes_1024x1024_160k.yml')
register_model_info({
    'model_name': 'bisenetv2',
    'suite': 'Seg',
    'config_path': BISENETV2_CFG_PATH,
    'auto_compression_config_path': BISENETV2_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
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
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
DEEPLABV3P_R101_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'deeplabv3p',
    'deeplabv3p_resnet101_os8_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'deeplabv3p_r101',
    'suite': 'Seg',
    'config_path': DEEPLABV3P_R101_CFG_PATH,
    'auto_compression_config_path': DEEPLABV3P_R101_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# Fast-SCNN
FASTSCNN_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'fastscnn',
                             'fastscnn_cityscapes_1024x1024_160k.yml')
register_model_info({
    'model_name': 'fastscnn',
    'suite': 'Seg',
    'config_path': FASTSCNN_CFG_PATH,
    'auto_compression_config_path': FASTSCNN_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# FCN
FCN_HRNETW18_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'fcn',
                                 'fcn_hrnetw18_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'fcn_hrnetw18',
    'suite': 'Seg',
    'config_path': FCN_HRNETW18_CFG_PATH,
    'auto_compression_config_path': FCN_HRNETW18_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
FCN_HRNETW48_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'fcn',
                                 'fcn_hrnetw48_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'fcn_hrnetw48',
    'suite': 'Seg',
    'config_path': FCN_HRNETW48_CFG_PATH,
    'auto_compression_config_path': FCN_HRNETW48_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
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
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
MOBILESEG_MV3_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'mobileseg',
    'mobileseg_mobilenetv3_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'mobileseg_mv3',
    'suite': 'Seg',
    'config_path': MOBILESEG_MV3_CFG_PATH,
    'auto_compression_config_path': MOBILESEG_MV3_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# OCRNet
OCRNET_HRNETW18_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'ocrnet',
    'ocrnet_hrnetw18_cityscapes_1024x512_160k.yml')
register_model_info({
    'model_name': 'ocrnet_hrnetw18',
    'suite': 'Seg',
    'config_path': OCRNET_HRNETW18_CFG_PATH,
    'auto_compression_config_path': OCRNET_HRNETW18_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
OCRNET_HRNETW48_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'ocrnet',
    'ocrnet_hrnetw48_cityscapes_1024x512_160k.yml')
register_model_info({
    'model_name': 'ocrnet_hrnetw48',
    'suite': 'Seg',
    'config_path': OCRNET_HRNETW48_CFG_PATH,
    'auto_compression_config_path': OCRNET_HRNETW48_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
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
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# PP-HumanSeg V2
PP_HUMANSEGV2_LITE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'contrib', 'PP-HumanSeg',
                                       'configs',
                                       'human_pp_humansegv2_lite.yml')
register_model_info({
    'model_name': 'pp_humansegv2_lite',
    'suite': 'Seg',
    'config_path': PP_HUMANSEGV2_LITE_CFG_PATH,
    'auto_compression_config_path': PP_HUMANSEGV2_LITE_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
PP_HUMANSEGV2_MOBILE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'contrib',
                                         'PP-HumanSeg', 'configs',
                                         'human_pp_humansegv2_mobile.yml')
register_model_info({
    'model_name': 'pp_humansegv2_mobile',
    'suite': 'Seg',
    'config_path': PP_HUMANSEGV2_MOBILE_CFG_PATH,
    'auto_compression_config_path': PP_HUMANSEGV2_MOBILE_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
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
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression'],
    'supported_train_opts': {
        'device': ['cpu', 'gpu_nxcx'],
        'dy2st': True,
        'amp': ['O1', 'O2']
    },
    'supported_evaluate_opts': {
        'device': ['cpu', 'gpu_nxcx']
    },
    'supported_predict_opts': {
        'device': ['cpu', 'gpu']
    },
    'supported_infer_opts': {
        'device': ['cpu', 'gpu']
    },
    'supported_compression_opts': {
        'device': ['cpu', 'gpu_nxcx']
    },
    'supported_dataset_types': []
})
PP_LITESEG_STDC2_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'pp_liteseg',
    'pp_liteseg_stdc2_cityscapes_1024x512_scale1.0_160k.yml')
register_model_info({
    'model_name': 'pp_liteseg_stdc2',
    'suite': 'Seg',
    'config_path': PP_LITESEG_STDC2_CFG_PATH,
    'auto_compression_config_path': PP_LITESEG_STDC2_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# PSPNet
PSPNET_R50_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'pspnet',
    'pspnet_resnet50_os8_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'pspnet_r50',
    'suite': 'Seg',
    'config_path': PSPNET_R50_CFG_PATH,
    'auto_compression_config_path': PSPNET_R50_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
PSPNET_R101_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'configs', 'pspnet',
    'pspnet_resnet101_os8_cityscapes_1024x512_80k.yml')
register_model_info({
    'model_name': 'pspnet_r101',
    'suite': 'Seg',
    'config_path': PSPNET_R101_CFG_PATH,
    'auto_compression_config_path': PSPNET_R101_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# RTFormer
RTFORMER_BASE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'rtformer',
                                  'rtformer_base_cityscapes_1024x512_120k.yml')
register_model_info({
    'model_name': 'rtformer_base',
    'suite': 'Seg',
    'config_path': RTFORMER_BASE_CFG_PATH,
    'auto_compression_config_path': RTFORMER_BASE_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# SegFormer
SEGFORMER_B0_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'segformer',
                                 'segformer_b0_cityscapes_1024x512_160k.yml')
register_model_info({
    'model_name': 'segformer_b0',
    'suite': 'Seg',
    'config_path': SEGFORMER_B0_CFG_PATH,
    'auto_compression_config_path': SEGFORMER_B0_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# TopFormer
TOPFORMER_BASE_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'topformer',
                                   'topformer_base_ade20k_512x512_160k.yml')
register_model_info({
    'model_name': 'topformer_base',
    'suite': 'Seg',
    'config_path': TOPFORMER_BASE_CFG_PATH,
    'auto_compression_config_path': TOPFORMER_BASE_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
TOPFORMER_SMALL_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'topformer',
                                    'topformer_small_ade20k_512x512_160k.yml')
register_model_info({
    'model_name': 'topformer_small',
    'suite': 'Seg',
    'config_path': TOPFORMER_SMALL_CFG_PATH,
    'auto_compression_config_path': TOPFORMER_SMALL_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
TOPFORMER_TINY_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'topformer',
                                   'topformer_tiny_ade20k_512x512_160k.yml')
register_model_info({
    'model_name': 'topformer_tiny',
    'suite': 'Seg',
    'config_path': TOPFORMER_TINY_CFG_PATH,
    'auto_compression_config_path': TOPFORMER_TINY_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})

# UNet
UNET_CFG_PATH = osp.join(REPO_ROOT_PATH, 'configs', 'unet',
                         'unet_cityscapes_1024x512_160k.yml')
register_model_info({
    'model_name': 'unet',
    'suite': 'Seg',
    'config_path': UNET_CFG_PATH,
    'auto_compression_config_path': UNET_CFG_PATH,
    'supported_apis':
    ['train', 'evaluate', 'predict', 'export', 'infer', 'compression']
})
