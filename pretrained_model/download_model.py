# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(LOCAL_PATH, "..", "test")
sys.path.append(TEST_PATH)

from test_utils import download_file_and_uncompress

model_urls = {
    # ImageNet Pretrained
    "mobilenetv2-2-0_bn_imagenet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar",
    "mobilenetv2-1-5_bn_imagenet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar",
    "mobilenetv2-1-0_bn_imagenet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar",
    "mobilenetv2-0-5_bn_imagenet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar",
    "mobilenetv2-0-25_bn_imagenet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar",
    "xception41_imagenet":
    "https://paddleseg.bj.bcebos.com/models/Xception41_pretrained.tgz",
    "xception65_imagenet":
    "https://paddleseg.bj.bcebos.com/models/Xception65_pretrained.tgz",
    "hrnet_w18_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w18_imagenet.tar",
    "hrnet_w30_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w30_imagenet.tar",
    "hrnet_w32_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w32_imagenet.tar",
    "hrnet_w40_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w40_imagenet.tar",
    "hrnet_w44_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w44_imagenet.tar",
    "hrnet_w48_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w48_imagenet.tar",
    "hrnet_w64_bn_imagenet":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w64_imagenet.tar",

    # COCO pretrained
    "deeplabv3p_mobilenetv2-1-0_bn_coco":
    "https://paddleseg.bj.bcebos.com/deeplab_mobilenet_x1_0_coco.tgz",
    "deeplabv3p_xception65_bn_coco":
    "https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz",
    "unet_bn_coco":
    "https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz",
    "pspnet50_bn_coco":
    "https://paddleseg.bj.bcebos.com/models/pspnet50_coco.tgz",
    "pspnet101_bn_coco":
    "https://paddleseg.bj.bcebos.com/models/pspnet101_coco.tgz",

    # Cityscapes pretrained
    "deeplabv3p_mobilenetv2-1-0_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz",
    "deeplabv3p_xception65_gn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/deeplabv3p_xception65_cityscapes.tgz",
    "deeplabv3p_xception65_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz",
    "unet_bn_coco":
    "https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz",
    "icnet_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/icnet_cityscapes.tar.gz",
    "pspnet50_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/pspnet50_cityscapes.tgz",
    "pspnet101_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/pspnet101_cityscapes.tgz",
    "hrnet_w18_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz",
    "fast_scnn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar",
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:\n  python download_model.py ${MODEL_NAME}")
        exit(1)

    model_name = sys.argv[1]
    if not model_name in model_urls.keys():
        print("Only support: \n  {}".format("\n  ".join(
            list(model_urls.keys()))))
        exit(1)

    url = model_urls[model_name]
    download_file_and_uncompress(
        url=url,
        savepath=LOCAL_PATH,
        extrapath=LOCAL_PATH,
        extraname=model_name)

    print("Pretrained Model download success!")
