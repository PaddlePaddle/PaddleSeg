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

import numpy as np
import cv2
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.core.predict import partition_list
from paddleseg.transforms import Compose
from paddleseg.utils import progbar

from qinspector.utils.logger import setup_logger

logger = setup_logger('SegPredictor')


class SegPredictor(object):
    def __init__(self, seg_config, seg_model):
        self.seg_config = seg_config
        self.model = seg_config.model
        self.model_path = seg_model
        self.transforms = Compose(seg_config.val_transforms)
        utils.utils.load_entire_model(self.model, self.model_path)
        self.model.eval()

    def postprocess(self, pred, img_data):
        result = []
        class_list = np.unique(pred)
        for cls_id in class_list:
            if cls_id == 0:
                continue  # skip background
            class_map = np.equal(pred, cls_id).astype(np.uint8)
            y_indices, x_indices = np.nonzero(class_map)
            y_min = int(y_indices.min())
            y_max = int(y_indices.max())
            x_min = int(x_indices.min())
            x_max = int(x_indices.max())

            contours, _ = cv2.findContours(class_map, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            for j in range(len(contours)):
                if len(contours[j]) <= 4:
                    continue
                polygon.append(contours[j].flatten().tolist())

            result.append({
                'image_path': img_data,
                'category_id': int(cls_id),
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                #'mask': class_map,
                'polygon': polygon,
                'area': int(np.sum(class_map > 0)),
                'isNG': 1
            })

        return result

    def roi_postprocess(self, pred, img_data):
        class_list = np.unique(pred)
        result = []
        for cls_id in class_list:
            if cls_id == 0:
                continue  # skip background
            class_map = np.equal(pred, cls_id).astype(np.uint8)

            crop_bbox = img_data['crop_bbox']
            bbox = img_data['bbox']
            # get mask
            offset_left = bbox[0] - crop_bbox[0]
            offset_top = bbox[1] - crop_bbox[1]
            offset_right = offset_left + bbox[2]
            offset_bottom = offset_top + bbox[3]
            class_map = class_map[int(offset_top):int(offset_bottom), int(
                offset_left):int(offset_right)].astype(np.uint8)
            contours, _ = cv2.findContours(class_map, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            for j in range(len(contours)):
                if len(contours[j]) <= 4:
                    continue
                contours[j][..., 0] += int(bbox[0])
                contours[j][..., 1] += int(bbox[1])
                polygon.append(contours[j].flatten().tolist())
            img_data.pop('img', None)
            img_data.pop('trans_info', None)
            img_data.pop('img_shape', None)
            img_data['polygon'] = polygon
            img_data['area'] = int(np.sum(class_map > 0))
            img_data['isNG'] = 1
            result.append(img_data)
        return result

    def preprocess(self, im_data):
        if not isinstance(im_data, dict):
            data = {}
            data['img'] = im_data
        else:
            data = im_data
        data = self.transforms(data)
        data['img'] = data['img'][np.newaxis, ...]
        data['img'] = paddle.to_tensor(data['img'])
        return data

    def predict(self,
                image_list,
                aug_pred=False,
                scales=1.0,
                flip_horizontal=True,
                flip_vertical=False,
                is_slide=False,
                stride=None,
                crop_size=None):

        results = []
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = partition_list(image_list, nranks)
        else:
            img_lists = [image_list]

        logger.info("Start to predict...")
        progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
        with paddle.no_grad():
            for i, im_data in enumerate(img_lists[local_rank]):
                data = self.preprocess(im_data)
                if aug_pred:
                    pred, _ = infer.aug_inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred, _ = infer.inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')

                if isinstance(im_data, dict):
                    result = self.roi_postprocess(pred, im_data)
                else:
                    result = self.postprocess(pred, im_data)
                results.extend(result)
                progbar_pred.update(i + 1)

        return results
