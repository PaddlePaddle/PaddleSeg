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

from ppdet.core.workspace import load_config
from qinspector.cvlib.workspace import register
from qinspector.det.engine import Predictor


@register
class Detection(object):
    def __init__(self, model_cfg, env_cfg):
        super(Detection, self).__init__()
        self.model_cfg = model_cfg
        det_config = model_cfg['config_path']
        det_model = model_cfg['model_path']
        det_config = load_config(det_config)
        self.score_threshold = 0.0
        if 'score_threshold' in self.model_cfg.keys():
            self.score_threshold = self.model_cfg['score_threshold']

        self.predictor = Predictor(det_config, mode='test')
        self.predictor.load_weights(det_model)

    def __call__(self, input):
        results = self.predictor.predict(
            input, score_thresh=self.score_threshold)
        return results
