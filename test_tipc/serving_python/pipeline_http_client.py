# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import requests
import json
import cv2
import base64
import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--img_path",
        help="The path of image.",
        type=str, )
    parser.add_argument(
        "--input_name",
        help="The input name of inference model.",
        type=str,
        default="x")
    return parser.parse_args()


def cv2_to_base64(image):
    """cv2_to_base64

    Convert an numpy array to a base64 object.

    Args:
        image: Input array.

    Returns: Base64 output of the input.
    """
    return base64.b64encode(image).decode('utf8')


if __name__ == "__main__":
    args = parse_args()
    img_path = args.img_path
    input_name = args.input_name
    url = "http://127.0.0.1:18080/seg/prediction"
    logid = 10000

    with open(img_path, 'rb') as file:
        image_data = file.read()
    # data should be transformed to the base64 format
    image = cv2_to_base64(image_data)
    data = {"key": [input_name], "value": [image], "logid": logid}
    # send requests
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())
