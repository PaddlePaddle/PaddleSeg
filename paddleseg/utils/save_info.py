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

import os
import json
import yaml

from paddleseg.utils import logger
from paddleseg.cvlibs import Config


def save_model_info(model_info, save_path):
    """
    save model info to the target path
    """
    with open(os.path.join(save_path, 'model.info.json'), 'w') as f:
        json.dump(model_info, f)
    logger.info("Already save model info in {}".format(save_path))


def update_train_results(args,
                         prefix,
                         metric_info,
                         done_flag=False,
                         last_num=5,
                         ema=False):
    assert last_num >= 1
    cfg = Config(args.config)
    train_results_path = os.path.join(args.save_dir, "train_results.json")
    save_model_tag = ["pdparams", "pdopt", "pdstates"]
    save_inference_tag = [
        "inference_config", "pdmodel", "pdiparams", "pdiparams.info"
    ]
    if ema:
        save_model_tag.append("pdema")
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = cfg.dic.get("pdx_model_name", "")
        train_results["label_dict"] = ""
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if "best_model" in prefix:
        train_results["models"]["best"]["score"] = metric_info["mIoU"]
        for tag in save_model_tag:
            if tag == "pdopt":
                continue
            train_results["models"]["best"][tag] = os.path.join(
                prefix, f"model.{tag}")
        for tag in save_inference_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"].copy()
        train_results["models"][f"last_{1}"]["score"] = metric_info["mIoU"]
        for tag in save_model_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, f"model.{tag}")
        for tag in save_inference_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
