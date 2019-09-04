# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(LOCAL_PATH, "..", "test")
sys.path.append(TEST_PATH)

from test_utils import download_file_and_uncompress

model_urls = {
    "deeplabv3plus_mobilenetv2-1-0_bn_cityscapes":
    "https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz",
    "unet_bn_coco": "https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz"
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
