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

import io
import os
import sys
import base64
import argparse

import yaml
import numpy as np
from PIL import Image

from paddle_serving_server.web_service import WebService, Op
from preprocess_ops import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        help="The path of config file.",
        type=str, )
    parser.add_argument(
        "--opt",
        help="Use --opt op.seg.local_service_conf.devices=xx to set devices.",
        type=str, )
    parser.add_argument(
        "--input_name",
        help="The output name of inference model.",
        type=str,
        default="x")
    parser.add_argument(
        "--output_name",
        help="The output name of inference model.",
        type=str,
        default="argmax_0.tmp_0")
    return parser.parse_args()


def read_update_cfg(config_path, opt):
    assert os.path.exists(config_path), "{} is not existed.".format(config_path)
    assert opt.startswith("op.seg.local_service_conf.devices"), \
        "Only support using opt to set op.seg.local_service_conf.devices"

    with io.open(config_path, encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)

    devices = opt.replace("op.seg.local_service_conf.devices=", "")
    if devices == "":
        devices = None
    config["op"]["seg"]["local_service_conf"]["devices"] = devices
    return config


class SegOp(Op):
    def init_op(self):
        self.seq = Compose([
            NormalizeImage(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToCHW()
        ])

    def set_in_out_name(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            byte_stream = io.BytesIO(data)
            img = Image.open(byte_stream)
            img = img.convert("RGB")
            img = self.seq(img)
            imgs.append(img[np.newaxis, :].copy())
        imgs = np.concatenate(imgs, axis=0)
        return {self.input_name: imgs}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        out = {"res_mean_val": []}
        results = fetch_dict[self.output_name]
        for val in results:
            val = val.tolist()
            val = np.array(val)
            out["res_mean_val"].append(np.mean(val))
        out["res_mean_val"] = str(out["res_mean_val"])
        return out, None, ""


class SegService(WebService):
    def __init__(self, name, input_name, output_name):
        super().__init__(name)
        self.input_name = input_name
        self.output_name = output_name

    def get_pipeline_response(self, read_op):
        seg_op = SegOp(name="seg", input_ops=[read_op])
        seg_op.set_in_out_name(self.input_name, self.output_name)
        return seg_op


if __name__ == '__main__':
    args = parse_args()
    config = read_update_cfg(args.config, args.opt)
    print("config:", config)

    uci_service = SegService("seg", args.input_name, args.output_name)
    uci_service.prepare_pipeline_config(yml_dict=config)
    uci_service.run_service()
