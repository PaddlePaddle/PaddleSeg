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
from .pspnet import *
from .gscnn import GSCNN
from .unet import UNet
from .hardnet import HarDNet
from .u2net import U2Net, U2Netp
from .attention_unet import AttentionUNet
from .unet_plusplus import UNetPlusPlus
from .unet_3plus import UNet3Plus
from .decoupled_segnet import DecoupledSegNet
from .emanet import *
from .isanet import *
from .dnlnet import *
from .setr import *
from .sfnet import *
from .pphumanseg_lite import *
from .mla_transformer import MLATransformer
from .portraitnet import PortraitNet
from .stdcseg import STDCSeg
from .segformer import SegFormer
from .pointrend import PointRend
from .ginet import GINet
from .segmenter import *
from .segnet import SegNet
from .encnet import ENCNet
from .hrnet_contrast import HRNetW48Contrast
from .espnet import ESPNetV2
from .dmnet import DMNet
from .espnetv1 import ESPNetV1
from .enet import ENet
from .bisenetv1 import BiseNetV1
from .fastfcn import FastFCN
from .pfpnnet import PFPNNet