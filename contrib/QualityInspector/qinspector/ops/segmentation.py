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

import cv2
from paddleseg.cvlibs import Config, SegBuilder

from qinspector.cvlib.workspace import register
from qinspector.utils.bbox_utils import adjust_bbox
from qinspector.seg.engine import SegPredictor


@register
class BaseSegmentation(object):
    def __init__(self, model_cfg, env_cfg=None):
        super(BaseSegmentation, self).__init__()
        seg_config = model_cfg['config_path']
        seg_config = Config(seg_config)
        builder = SegBuilder(seg_config)
        seg_model = model_cfg['model_path']
        self.aug_pred = model_cfg.get('aug_pred', False)
        self.predictor = SegPredictor(builder, seg_model)

    def __call__(self, inputs):
        results = self.predictor.predict(inputs, aug_pred=self.aug_pred)
        return results


@register
class CropSegmentation(object):
    def __init__(self, model_cfg, env_cfg=None):
        super(CropSegmentation, self).__init__()
        seg_config = model_cfg['config_path']
        seg_config = Config(seg_config)
        builder = SegBuilder(seg_config)
        seg_model = model_cfg['model_path']
        self.pad_scale = model_cfg.get('pad_scale', 0.5)
        self.aug_pred = model_cfg.get('aug_pred', False)
        self.predictor = SegPredictor(builder, seg_model)

    def __call__(self, input):
        for data in input:
            image_path = data['image_path']
            bbox = list(map(int, data['bbox']))
            img = cv2.imread(image_path)
            crop_bbox = adjust_bbox(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                img_shape=img.shape[:2],
                pad_scale=self.pad_scale)
            img_crop = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[
                2], :]
            data['img'] = img_crop
            data['img_shape'] = img.shape[:2]
            data['crop_bbox'] = [
                crop_bbox[0], crop_bbox[1], crop_bbox[2] - crop_bbox[0],
                crop_bbox[3] - crop_bbox[1]
            ]
        results = self.predictor.predict(input, aug_pred=self.aug_pred)
        return results
