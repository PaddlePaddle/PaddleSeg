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

from uapi import PaddleModel, Config

if __name__ == '__main__':
    model_name = 'pphumanseg_lite'

    model = PaddleModel(model_name=model_name)

    # Hard-code paths
    save_dir = f"uapi_demo/output"
    dataset_dir = "uapi_demo/data/mini_supervisely"
    infer_input_path = "uapi_demo/data/mini_supervisely/Images/baby-boy-hat-covered-101537.png"

    weight_path = osp.join(save_dir, "iter_10", "model.pdparams")
    export_dir = osp.join(save_dir, 'infer')
    pred_save_dir = osp.join(save_dir, 'pred_res')
    infer_save_dir = osp.join(save_dir, 'infer_res')
    compress_save_dir = osp.join(save_dir, 'compress')

    # Do test
    model.train(
        dataset=dataset_dir,
        batch_size=4,
        epochs_iters=50,
        device='gpu:0',
        amp='O1',
        dy2st=True,
        save_dir=save_dir)

    model.evaluate(dataset=dataset_dir, device='gpu', amp='O2')

    model.predict(
        weight_path=weight_path,
        device='gpu',
        input_path=infer_input_path,
        save_dir=pred_save_dir)

    model.export(weight_path=weight_path, input_shape=None, save_dir=export_dir)

    model.infer(
        model_dir=export_dir,
        device='gpu',
        input_path=infer_input_path,
        save_dir=infer_save_dir)

    model.compression(
        dataset=dataset_dir,
        batch_size=2,
        epochs_iters=10,
        device='gpu',
        weight_path=weight_path,
        use_vdl=False,
        save_dir=compress_save_dir,
        input_shape=None)
