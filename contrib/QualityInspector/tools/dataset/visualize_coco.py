# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import os.path as osp

from PIL import Image, ImageDraw
from pycocotools.coco import COCO


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path', type=str, help="The path of coco format json file.")
    parser.add_argument(
        '--root_path',
        type=str,
        default="",
        help="The directory of images, default None if the path of images is absolute path in json file."
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="./show/",
        help="The directory for saving visualization result.")
    return parser.parse_args()


def visualization_bbox(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)

    coco = COCO(args.json_path)
    coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
    img_ids = list(sorted(coco.imgs.keys()))
    for img_id in img_ids:
        im_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        annos = coco.loadAnns(ann_ids)
        #coco.showAnns(annos)
        if len(annos) == 0:
            continue
        image = Image.open(os.path.join(args.root_path, im_info[
            'file_name'])).convert('RGB')
        draw = ImageDraw.Draw(image)
        for anno in annos:
            x, y, w, h = anno["bbox"]
            draw.line(
                [(x, y), (x, int(y + h)), (int(x + w), int(y + h)),
                 (int(x + w), y), (x, y)],
                width=2,
                fill=(255, 0, 0))
            text = "class: {}".format(str(coco_classes[anno["category_id"]]))
            _, th = draw.textsize(text)
            draw.text((x, y - th), text)
        image.save(osp.join(args.save_path, osp.basename(im_info['file_name'])))


if __name__ == "__main__":
    args = parse_args()
    visualization_bbox(args)
