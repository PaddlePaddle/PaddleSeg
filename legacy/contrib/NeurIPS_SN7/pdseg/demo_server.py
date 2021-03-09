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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from flask import Flask
from flask import request

# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import numpy as np
from PIL import Image
import paddle.fluid as fluid

from utils.config import cfg
cfg.update_from_file("demo_server.yaml")
from models.model_builder import build_model
from models.model_builder import ModelPhase

server = Flask(__name__)


# ###### global init ######
def load_model():
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    pred, logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)
    fetch_list = [pred.name]
    # Clone forward graph
    test_prog = test_prog.clone(for_test=True)

    # Get device environment
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    ckpt_dir = cfg.TEST.TEST_MODEL
    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        try:
            fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
        except:
            fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)

    # # Get device environment
    # places = [fluid.CUDAPlace(i) for i in range(4)]
    # exes = [fluid.Executor(places[i]) for i in range(4)]
    # for exe in exes:
    #     exe.run(startup_prog)
    #
    # ckpt_dir = cfg.TEST.TEST_MODEL
    # if ckpt_dir is not None:
    #     print('load test model:', ckpt_dir)
    #     for i in range(4):
    #         try:
    #             fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exes[i])
    #         except:
    #             fluid.io.load_params(exes[i], ckpt_dir, main_program=test_prog)

    return fetch_list, test_prog, exe  #s


fetch_list_diff, test_prog_diff, exe_diff = load_model()

cfg.DATASET.INPUT_IMAGE_NUM = 1
fetch_list_seg, test_prog_seg, exe_seg = load_model()


# ###### inference ######
def normalize_image(img):
    img = img.transpose((2, 0, 1)).astype('float32') / 255.0
    img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
    img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))
    img -= img_mean
    img /= img_std
    return img


def inference(img1, img2, fetch_list, test_prog, exe):
    img1 = normalize_image(img1)[np.newaxis, :, :, :]
    if img2 is not None:
        img2 = normalize_image(img2)[np.newaxis, :, :, :]
        pred, = exe.run(
            program=test_prog,
            feed={
                'image1': img1,
                'image2': img2
            },
            fetch_list=fetch_list,
            return_numpy=True)

        idx = pred.shape[0] // cfg.DATASET.INPUT_IMAGE_NUM
        pred1, pred2 = pred[:idx], pred[
            idx:]  # fluid.layers.split(pred, 2, dim=0)
        res_map1 = np.squeeze(pred1[0, :, :, :]).astype(np.uint8)
        res_map2 = np.squeeze(pred2[0, :, :, :]).astype(np.uint8)

        unchange_idx = np.where((res_map1 - res_map2) == 0)
        diff = res_map1 * 8 + res_map2
        diff[unchange_idx] = 0
        return res_map1, res_map2, diff
    else:
        pred, = exe.run(
            program=test_prog,
            feed={'image1': img1},
            fetch_list=fetch_list,
            return_numpy=True)
        res_map = np.squeeze(pred[0, :, :, :]).astype(np.uint8)
        return res_map


@server.route('/put_image', methods=['GET', 'POST'])
def listen():
    data = json.loads(request.data)
    img1 = np.array(data['img1'])
    img2 = None
    if 'img2' in data:
        img2 = np.array(data['img2'])
        cfg.DATASET.INPUT_IMAGE_NUM = 2

        res_map1, res_map2, diff = inference(
            img1, img2, fetch_list_diff, test_prog_diff, exe_diff)  #s[idx % 4])
        ret = {
            "res_map1": res_map1.tolist(),
            "res_map2": res_map2.tolist(),
            "diff": diff.tolist(),
        }
    else:
        cfg.DATASET.INPUT_IMAGE_NUM = 1
        res_map = inference(img1, img2, fetch_list_seg, test_prog_seg, exe_seg)
        ret = {"res_map1": res_map.tolist()}
    return json.dumps(ret)


def main():
    server.run(host='10.255.94.19', port=8000, debug=True)


if __name__ == '__main__':
    main()
