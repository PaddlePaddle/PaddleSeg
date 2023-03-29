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
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import cv2
from pycocotools.coco import COCO

from qinspector.utils.bbox_utils import adjust_bbox
from qinspector.utils.visualizer import polygons_to_bitmask


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoCo format Json convert to RoI binary Mask.')
    # Parameters
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='The path of coco format json file.')
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='The directory of images, default None if the path of images is absolute path in json file.'
    )
    parser.add_argument(
        '--seg_classid',
        type=int,
        nargs='+',
        default=None,
        help='Classid for converting to RoI, default None if all classes need to be converted.'
    )
    parser.add_argument(
        '--pad_scale',
        type=float,
        default=0.5,
        help='The padding scale of box to crop image.')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.png',
        help='The suffix of filename between gt and image.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/RoI/',
        help='The directory for saving the RoI images, masks and list.txt.')
    return parser.parse_args()


def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)


def generate_mask_RoI(data,
                      image_root,
                      output_path,
                      suffix,
                      class_id=None,
                      pad_scale=0.5):
    output_image_path = osp.join(output_path, 'images')
    _mkdir_p(output_image_path)
    output_anno_path = osp.join(output_path, 'anno')
    _mkdir_p(output_anno_path)
    file_list = osp.join(output_path, 'RoI.txt')
    f = open(file_list, "w")

    if not class_id:
        class_id = data.getCatIds()
    img_ids = list(sorted(data.imgs.keys()))
    for img_id in img_ids:
        im_info = data.loadImgs(img_id)[0]
        ann_ids = data.getAnnIds(imgIds=img_id, catIds=class_id, iscrowd=False)
        annos = data.loadAnns(ann_ids)
        if len(annos) == 0:
            continue
        img = cv2.imread(osp.join(image_root, im_info['file_name']))

        if img is None:
            raise FileNotFoundError('{} is not found.'.format(
                osp.join(image_root, im_info['file_name'])))

        base_name = os.path.basename(im_info['file_name']).split('.')[0]
        polygons = []
        for anno in annos:
            polygons.extend(anno['segmentation'])
        mask = polygons_to_bitmask(polygons, img.shape[0], img.shape[1])
        for idx, anno in enumerate(annos):
            bbox = list(map(int, anno['bbox']))
            bbox = adjust_bbox(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                img.shape[:2],
                pad_scale=pad_scale)
            crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            crop_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_save_path = osp.join(output_image_path,
                                     f'{base_name}_{idx}.png')
            anno_save_path = osp.join(output_anno_path,
                                      f'{base_name}_{idx}{suffix}')
            cv2.imwrite(img_save_path, crop_img)
            cv2.imwrite(anno_save_path, crop_mask)
            line = img_save_path + " " + anno_save_path + '\n'
            f.write(line)
    f.close()


if __name__ == '__main__':
    args = parse_args()
    data = COCO(args.json_path)
    generate_mask_RoI(data, args.image_path, args.output_path, args.suffix,
                      args.seg_classid, args.pad_scale)
