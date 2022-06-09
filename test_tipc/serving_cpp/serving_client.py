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

import sys
import base64
import io
import argparse

import numpy as np
from PIL import Image
from paddle_serving_client import Client


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--img_path",
        help="The path of image.",
        type=str,
        default="../data/cityscapes_small.png")
    parser.add_argument(
        "--input_name",
        help="The input name of inference model.",
        type=str,
        default="x")
    parser.add_argument(
        "--output_name",
        help="The output name of inference model.",
        type=str,
        default="argmax_0.tmp_0")
    return parser.parse_args()


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


def preprocess(img_file, input_name, output_name):
    with open(img_file, 'rb') as file:
        image_data = file.read()
    image = cv2_to_base64(image_data)
    feed = {input_name: image}
    fetch = [output_name]
    return feed, fetch


def postprocess(fetch_map, output_name):
    fetch_dict = {"res_mean_val": []}
    for val in fetch_map[output_name]:
        val = val.tolist()
        val = np.array(val)
        fetch_dict["res_mean_val"].append(np.mean(val))
    fetch_dict["res_mean_val"] = str(fetch_dict["res_mean_val"])
    return fetch_dict


if __name__ == "__main__":
    url = "127.0.0.1:9997"
    logid = 10000

    args = parse_args()
    img_path = args.img_path
    input_name = args.input_name
    output_name = args.output_name

    client = Client()
    client.load_client_config("serving_client/serving_client_conf.prototxt")
    client.connect([url])

    feed, fetch = preprocess(img_path, input_name, output_name)
    fetch_map = client.predict(feed=feed, fetch=fetch)
    result = postprocess(fetch_map, output_name)
    print("result:", result, "\n")
