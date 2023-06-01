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

from qinspector.cvlib.workspace import create, register
from qinspector.utils.logger import setup_logger

logger = setup_logger('PostProcess')


@register
class PostProcess(object):
    def __init__(self, model_cfg, env_cfg=None):
        super(PostProcess, self).__init__()
        self.env_cfg = env_cfg
        self.op_name2op = {}
        self.rule = []
        self.init(model_cfg)

    def init(self, cfg):
        self.rules = []
        if isinstance(cfg, list):
            for sub_op in cfg:
                sub_op_arch = list(sub_op.keys())[0]
                sub_op_cfg = list(sub_op.values())[0]
                sub_op = create(sub_op_arch, sub_op_cfg, self.env_cfg)
                self.op_name2op[sub_op_arch] = sub_op
                self.rules.append(sub_op)
        logger.info("PostProcess has {} rules".format(len(self.rules)))

    def __len__(self):
        return len(self.rules)

    def __call__(self, input):
        for rule in self.rules:
            input = rule(input)
        return input


@register
class JudgeDetByScores(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeDetByScores, self).__init__()
        self.score_threshold = cfg['score_threshold']

    def __call__(self, inputs):
        for _, img_info in inputs.items():
            preds = img_info['pred']
            for pred in preds:
                if pred.get("isNG", 1):
                    if isinstance(self.score_threshold, dict):
                        if pred['category_id'] in self.score_threshold.keys():
                            threshold = self.score_threshold[pred[
                                'category_id']]
                        else:
                            pred['isNG'] = 1
                            continue
                    else:
                        threshold = self.score_threshold

                    if pred['score'] >= threshold:
                        pred['isNG'] = 1
                    else:
                        pred['isNG'] = 0
        return inputs


@register
class JudgeByLengthWidth(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeByLengthWidth, self).__init__()
        self.len_thresh = cfg['len_thresh']

    def __call__(self, inputs):
        for _, img_info in inputs.items():
            preds = img_info['pred']
            for pred in preds:
                if pred.get("isNG", 1):
                    if isinstance(self.len_thresh, dict):
                        if pred['category_id'] in self.len_thresh.keys():
                            threshold = self.len_thresh[pred['category_id']]
                        else:
                            pred['isNG'] = 1
                            continue
                    else:
                        threshold = self.len_thresh
                    if pred['bbox'][2] >= threshold or pred['bbox'][
                            3] >= threshold:
                        pred['isNG'] = 1
                    else:
                        pred['isNG'] = 0

        return inputs


@register
class JudgeByArea(object):
    def __init__(self, cfg, env_cfg=None):
        super(JudgeByArea, self).__init__()
        self.area_thresh = cfg['area_thresh']

    def __call__(self, inputs):
        for _, img_info in inputs.items():
            preds = img_info['pred']
            for pred in preds:
                if pred.get("isNG", 1):
                    if isinstance(self.area_thresh, dict):
                        if pred['category_id'] in self.area_thresh.keys():
                            threshold = self.area_thresh[pred['category_id']]
                        else:
                            pred['isNG'] = 1
                            continue
                    else:
                        threshold = self.area_thresh

                    if 'area' in pred.keys():
                        if pred['area'] >= threshold:
                            pred['isNG'] = 1
                        else:
                            pred['isNG'] = 0
                    elif 'bbox' in pred.keys():
                        if pred['bbox'][2] * pred['bbox'][3] >= threshold:
                            pred['isNG'] = 1
                        else:
                            pred['isNG'] = 0
        return inputs
