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


def download_pet_dataset(savepath, extrapath):
    url = "https://paddleseg.bj.bcebos.com/dataset/mini_pet.zip"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


def download_unet_coco_model(savepath, extrapath):
    url = "https://bj.bcebos.com/v1/paddleseg/models/unet_coco_init.tgz"
    download_file_and_uncompress(
        url=url, savepath=savepath, extrapath=extrapath)


if __name__ == "__main__":
    download_pet_dataset(LOCAL_PATH, DATASET_PATH)
    download_unet_coco_model(LOCAL_PATH, MODEL_PATH)

    model_name = "unet_pet"
    test_model = os.path.join(LOCAL_PATH, "models", "unet_coco_init")
    cfg = os.path.join(LOCAL_PATH, "..", "configs",
                       "{}.yaml".format(model_name))
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

    train(
        flags=["--cfg", cfg, "--use_gpu", "--log_steps", "10"],
        options=[
            "SOLVER.NUM_EPOCHS", "1", "TRAIN.PRETRAINED_MODEL_DIR", test_model,
            "TRAIN.MODEL_SAVE_DIR", saved_model, "DATASET.TRAIN_FILE_LIST",
            os.path.join(DATASET_PATH, "mini_pet", "file_list",
                         "train_list.txt"), "DATASET.VAL_FILE_LIST",
            os.path.join(DATASET_PATH, "mini_pet", "file_list",
                         "val_list.txt"), "DATASET.TEST_FILE_LIST",
            os.path.join(DATASET_PATH, "mini_pet", "file_list",
                         "test_list.txt"), "DATASET.DATA_DIR",
            os.path.join(DATASET_PATH, "mini_pet"), "BATCH_SIZE", "1"
        ],
        devices=devices)

    eval(
        flags=["--cfg", cfg, "--use_gpu"],
        options=[
            "TEST.TEST_MODEL",
            os.path.join(saved_model, "final"), "DATASET.VAL_FILE_LIST",
            os.path.join(DATASET_PATH, "mini_pet", "file_list", "val_list.txt"),
            "DATASET.DATA_DIR",
            os.path.join(DATASET_PATH, "mini_pet")
        ],
        devices=devices)

    vis(flags=["--cfg", cfg, "--use_gpu", "--local_test", "--vis_dir", vis_dir],
        options=[
            "DATASET.TEST_FILE_LIST",
            os.path.join(DATASET_PATH, "mini_pet", "file_list",
                         "test_list.txt"), "DATASET.DATA_DIR",
            os.path.join(DATASET_PATH, "mini_pet"), "TEST.TEST_MODEL",
            os.path.join(saved_model, "final")
        ],
        devices=devices)

    export_model(
        flags=["--cfg", cfg],
        options=[
            "TEST.TEST_MODEL",
            os.path.join(saved_model, "final"), "FREEZE.SAVE_DIR",
            freeze_save_dir
        ],
        devices=devices)
