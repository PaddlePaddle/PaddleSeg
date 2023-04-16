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

import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw
import pycocotools.mask as mask_util
from ppdet.utils.colormap import colormap


def polygons_to_bitmask(polygons, height, width):
    """converts a set of polygons into a bitmask """
    if any(len(p) <= 4 for p in polygons):
        polygons = [p for p in polygons if len(p) > 4]
    if len(polygons) == 0:
        return np.zeros((height, width)).astype(int)

    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(int)


def draw_bboxes(path, annos, clsid_to_name, output_dir=None):
    if len(annos) == 0:
        return
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for anno in annos:
        x, y, w, h = anno["bbox"]
        draw.line(
            [(x, y), (x, int(y + h)), (int(x + w), int(y + h)), (int(x + w), y),
             (x, y)],
            width=2,
            fill=(255, 0, 0))
        text = "class: {}".format(
            str(clsid_to_name[anno["category_id"]]['name']))
        _, th = draw.textsize(text)
        draw.text((x, y - th), text)
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    image.save(osp.join(output_dir, osp.basename(path)))


def show_result(results, output_dir=None, class_name=None):
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)

    catid2color = {}
    for im_path, preds in results.items():
        image = Image.open(im_path).convert("RGB")
        color_list = colormap(rgb=True)
        if len(preds['pred']) == 0:
            continue
        for pred in preds['pred']:
            draw = ImageDraw.Draw(image)
            cate_id = pred['category_id']
            if cate_id not in catid2color:
                idx = np.random.randint(len(color_list))
                catid2color[cate_id] = color_list[idx]
            color = tuple(catid2color[cate_id])

            if 'bbox' in pred:
                bbox = pred['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                draw.line(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                     (xmin, ymin)],
                    width=2,
                    fill=color)

                isNG = pred['isNG']
                if class_name:
                    cate_name = class_name
                else:
                    cate_name = pred.get('category_name', str(cate_id))
                if 'score' in pred:
                    score = pred['score']
                    text = "{} {:.2f} {}".format(cate_name, score, 'NG'
                                                 if isNG else 'OK')
                else:
                    text = "{} {}".format(cate_name, 'NG' if isNG else 'OK')

                tw, th = draw.textsize(text)
                if ymin - th >= 1:
                    draw.rectangle(
                        [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)],
                        fill=color)
                    draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
                else:
                    draw.rectangle(
                        [(xmin + 1, ymax), (xmin + tw + 1, ymax + th)],
                        fill=color)
                    draw.text((xmin + 1, ymax + 1), text, fill=(255, 255, 255))

            if 'polygon' in pred:
                polygons = pred['polygon']
                if len(polygons) == 0 or len(polygons[0]) < 4:
                    continue
                alpha = 0.7
                w_ratio = .4
                img_array = np.array(image).astype('float32')
                color = np.asarray(color)
                for c in range(3):
                    color[c] = color[c] * (1 - w_ratio) + w_ratio * 255

                mask = polygons_to_bitmask(polygons, img_array.shape[0],
                                           img_array.shape[1])
                idx = np.nonzero(mask)
                img_array[idx[0], idx[1], :] *= 1.0 - alpha
                img_array[idx[0], idx[1], :] += alpha * color
                image = Image.fromarray(img_array.astype('uint8'))

        im_file = os.path.basename(im_path)
        pred_saved_path = os.path.join(output_dir,
                                       os.path.splitext(im_file)[0] + ".png")
        image.save(pred_saved_path)
