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

from .unet import UNet
from .hrnet import *

MODELS = {
    "UNet": UNet,
    "HRNet_W18_Small_V1": HRNet_W18_Small_V1,
    "HRNet_W18_Small_V2": HRNet_W18_Small_V2,
    "HRNet_W18": HRNet_W18,
    "HRNet_W30": HRNet_W30,
    "HRNet_W32": HRNet_W32,
    "HRNet_W40": HRNet_W40,
    "HRNet_W44": HRNet_W44,
    "HRNet_W48": HRNet_W48,
    "HRNet_W60": HRNet_W48,
    "HRNet_W64": HRNet_W64,
    "SE_HRNet_W18_Small_V1": SE_HRNet_W18_Small_V1,
    "SE_HRNet_W18_Small_V2": SE_HRNet_W18_Small_V2,
    "SE_HRNet_W18": SE_HRNet_W18,
    "SE_HRNet_W30": SE_HRNet_W30,
    "SE_HRNet_W32": SE_HRNet_W30,
    "SE_HRNet_W40": SE_HRNet_W40,
    "SE_HRNet_W44": SE_HRNet_W44,
    "SE_HRNet_W48": SE_HRNet_W48,
    "SE_HRNet_W60": SE_HRNet_W60,
    "SE_HRNet_W64": SE_HRNet_W64
}
