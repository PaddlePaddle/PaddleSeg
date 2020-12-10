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

from test_utils import download_file_and_uncompress, train, eval, vis, export_model
import os
import argparse

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(LOCAL_PATH, "..", "dataset")
MODEL_PATH = os.path.join(LOCAL_PATH, "models")


def download_cityscapes_dataset(savepath, extrapath):
    url = "https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


def download_deeplabv3p_xception65_cityscapes_model(savepath, extrapath):
    url = "https://paddleseg.bj.bcebos.com/models/deeplabv3p_xception65_cityscapes.tgz"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


if __name__ == "__main__":
    download_cityscapes_dataset(".", DATASET_PATH)
    download_deeplabv3p_xception65_cityscapes_model(".", MODEL_PATH)

    model_name = "deeplabv3p_xception65_cityscapes"
    test_model = os.path.join(LOCAL_PATH, "models", model_name)
    cfg = os.path.join(LOCAL_PATH, "configs", "{}.yaml".format(model_name))
    freeze_save_dir = os.path.join(LOCAL_PATH, "inference_model", model_name)
    vis_dir = os.path.join(LOCAL_PATH, "visual", model_name)
    saved_model = os.path.join(LOCAL_PATH, "saved_model", model_name)

    parser = argparse.ArgumentParser(description="PaddleSeg loacl test")
    parser.add_argument(
        "--devices",
        dest="devices",
        help="GPU id of running. if more than one, use spacing to separate.",
        nargs="+",
        default=[0],
        type=int)
    args = parser.parse_args()

    devices = [str(x) for x in args.devices]

    export_model(
        flags=["--cfg", cfg],
        options=[
            "TEST.TEST_MODEL", test_model, "FREEZE.SAVE_DIR", freeze_save_dir
        ],
        devices=devices)

    # Final eval results should be #image=500 acc=0.9615 IoU=0.7804
    eval(
        flags=["--cfg", cfg, "--use_gpu"],
        options=["TEST.TEST_MODEL", test_model],
        devices=devices)

    vis(flags=["--cfg", cfg, "--use_gpu", "--local_test", "--vis_dir", vis_dir],
        options=["TEST.TEST_MODEL", test_model],
        devices=devices)

    train(
        flags=["--cfg", cfg, "--use_gpu", "--log_steps", "10"],
        options=[
            "SOLVER.NUM_EPOCHS", "1", "TRAIN.PRETRAINED_MODEL_DIR", test_model,
            "TRAIN.MODEL_SAVE_DIR", saved_model
        ],
        devices=devices)
