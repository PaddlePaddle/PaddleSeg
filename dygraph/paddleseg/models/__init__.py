# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .backbones import *
from .losses import *

from .ann import *
from .bisenet import *
from .danet import *
from .deeplab import *
from .fast_scnn import *
from .fcn import *
from .gcnet import *
from .ocrnet import *
from .ocrnet_nv import OCRNetNV
from .pspnet import *
from .unet import UNet
from .mscale_ocr_finetune import MscaleOCRFinetune
from .mscale_ocr_pretrain import MscaleOCRPretrain
from .mscale_ocrnet import MscaleOCRNet
from .ms_class_wise_ocrnet import MsClassWiseOCRNet
from .ms_2attention_ocrnet import Ms2AttentionOCRNet
from .ms_ocrnet_attention_coef import MsAttentionCoefOCRNet
from .ms_2input_atten_ocrnet import Ms2InputAttenOCRNet
from .ms_cls_input_atten_ocrnet import MsClsInputAttenOCRNet
from .ms_cls_input_no_aux_atten_ocrnet import MsClsInputNoAuxAttenOCRNet
from .ms_cls_input_atten_ocrnet_crp import MsClsInputAttenOCRNetCRP
