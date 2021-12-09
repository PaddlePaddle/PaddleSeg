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


import argparse
import glob
import json
import os
import shutil

# follow openMMDet path processing patterns
from pathlib import Path
import numpy as np
import PIL.ImageDraw

from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict

# @todo TODO(refactor)
settings = edict()
settings.TRAIN_PROPORTION = 0.8
settings.VALIDATION_PROPORTION = 0.2

useQtBottomLeft = True  # should be Qt TopLeft


class MyEncoder(json.JSONEncoder):
    def default(self, val):
        if isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        else:
            return super(MyEncoder, self).default(val)


def save_coco_annotation(dataset, coco_annotation_path):
    data_coco = {"images": [], "annotations": []}

    imgs_info_indexed_by_img_name = {}
    imgs_info_indexed_by_img_idx = {}

    print("Generating dataset from:", coco_annotation_path)
    with open(coco_annotation_path) as f:
        data = json.load(f)

        # fetch categroies, info and licenses fields
        data_coco["categories"] = data["categories"]
        data_coco["info"] = data["info"]
        data_coco["licenses"] = data["licenses"]

        # fetch images and annotations
        for img_info in data["images"]:
            img_name = img_info["file_name"]
            if imgs_info_indexed_by_img_name.get(img_name, None) is None:
                imgs_info_indexed_by_img_name[img_name] = {}
            imgs_info_indexed_by_img_name[img_name]["img_info"] = img_info
            imgs_info_indexed_by_img_idx[img_info["id"]] = img_info

        for annotation in data["annotations"]:
            img_idx = annotation["image_id"]
            img_info = imgs_info_indexed_by_img_idx[img_idx]
            img_name = img_info["file_name"]
            # annotation polygon
            epsilon = 1e-5
            if "iscrowd" not in annotation:
                annotation["iscrowd"] = 0
            if annotation["area"] < epsilon:
                annotation["area"] = annotation["bbox"][2] * annotation["bbox"][3]
            if useQtBottomLeft:
                annotation["bbox"][1] = annotation["bbox"][1] - annotation["bbox"][3]

            imgs_info_indexed_by_img_name[img_name]["img_annotation"] = annotation

    for img_path in dataset:
        img_file = Path(img_path)
        img_name = img_file.name
        img_info_with_annotation = imgs_info_indexed_by_img_name[img_name]
        data_coco["images"].append(img_info_with_annotation["img_info"])
        data_coco["annotations"].append(img_info_with_annotation["img_annotation"])

    return data_coco


# TODO: 图像没有对应标签略过
# TODO: bbox貌似计算的不对
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_type",
        default="eiseg",
        help="type type of dataset is supported is EISeg",
    )
    parser.add_argument("--json_input_dir", help="input annotation directory")
    parser.add_argument("--image_input_dir", help="image directory")
    parser.add_argument("--output_dir", help="output dataset directory")

    args = parser.parse_args()
    try:
        assert args.dataset_type in ["coco", "eiseg", "labelme"]
    except AssertionError as e:
        print("Only coco and EISeg dataset is supported for the moment!")
        return -1

    coco_annotation_dir = args.json_input_dir
    coco_annotation_file = os.path.join(coco_annotation_dir, "annotations.json")

    setattr(settings, "DATA_DIR", args.image_input_dir)
    raw_images = list(glob.iglob(os.path.join(settings.DATA_DIR, "*.jpg")))
    raw_images = sorted(raw_images, key=lambda x: os.path.split(x)[1].split(".")[0])

    assert settings.TRAIN_PROPORTION > 0 and settings.TRAIN_PROPORTION < 1
    assert settings.TRAIN_PROPORTION + settings.VALIDATION_PROPORTION == 1

    setattr(settings, "coco_dataset_path", args.output_dir)
    train_path = os.path.join(settings.coco_dataset_path, "train")
    val_path = os.path.join(settings.coco_dataset_path, "val")
    annotation_path = os.path.join(settings.coco_dataset_path, "annotations")
    # TODO: 路径部分存在
    if not os.path.exists(settings.coco_dataset_path):
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(annotation_path)

    # Disk task : move images and annotation files to coco files hierarchy
    train_dataset, val_dataset = train_test_split(
        raw_images, train_size=settings.TRAIN_PROPORTION
    )

    for img_path in train_dataset:
        img_name = Path(img_path).name  # similar to boot api
        shutil.copyfile(img_path, os.path.join(train_path, img_name))

    for img_path in val_dataset:
        img_name = Path(img_path).name  # similar to boot api
        shutil.copyfile(img_path, os.path.join(val_path, img_name))

    # deal with coco annotation file
    data_coco_train = save_coco_annotation(train_dataset, coco_annotation_file)
    train_json_path = os.path.join(annotation_path, "train.json")
    json.dump(data_coco_train, open(train_json_path, "w"), indent=4, cls=MyEncoder)

    data_coco_val = save_coco_annotation(val_dataset, coco_annotation_file)
    val_json_path = os.path.join(annotation_path, "val.json")
    json.dump(data_coco_val, open(val_json_path, "w"), indent=4, cls=MyEncoder)


if __name__ == "__main__":
    import sys

    sys.exit(main())
