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

import typing

from tqdm import tqdm
from ppdet.core.workspace import create
from ppdet.data.source.category import get_categories
from ppdet.engine import Trainer

from qinspector.utils.logger import setup_logger

logger = setup_logger('DetPredictor')


class Predictor(Trainer):
    def predict(self, images, score_thresh=0.0):

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()
        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)

        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)
            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()

            infer_res = self.get_det_res(
                outs['bbox'],
                outs['bbox_num'],
                outs['im_id'],
                clsid2catid,
                catid2name,
                imid2path,
                score_thresh=score_thresh)
            results.extend(infer_res)

        return results

    def get_det_res(self,
                    bboxes,
                    bbox_nums,
                    image_id,
                    label_to_cat_id_map,
                    catid2name,
                    imid2path,
                    bias=0,
                    score_thresh=0.0):
        det_res = []
        k = 0
        for i in range(len(bbox_nums)):
            cur_image_id = int(image_id[i][0])
            cur_image_path = imid2path[cur_image_id]
            det_nums = bbox_nums[i]
            for j in range(det_nums):
                dt = bboxes[k]
                k = k + 1
                num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                if int(num_id) < 0:
                    continue
                if score < score_thresh:
                    continue
                category_id = label_to_cat_id_map[int(num_id)]
                w = xmax - xmin + bias
                h = ymax - ymin + bias
                bbox = [xmin, ymin, w, h]
                dt_res = {
                    'image_id': cur_image_id,
                    'image_path': cur_image_path,
                    'category_id': category_id,
                    'category_name': catid2name[category_id],
                    'bbox': bbox,
                    'score': score,
                    'isNG': 1
                }
                det_res.append(dt_res)
        return det_res
