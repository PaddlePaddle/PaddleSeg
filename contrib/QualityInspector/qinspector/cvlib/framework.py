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

from qinspector.ops import __all__
from qinspector.cvlib.workspace import create


class Builder(object):
    """
    The executor which implements model series pipeline

    Args:
        env_cfg: The enrionment configuration
        model_cfg: The models configuration
    """

    def __init__(self, model_cfg, env_cfg=None):
        self.model_cfg = model_cfg
        self.op_name2op = {}
        self.has_output_op = False
        for op in model_cfg:
            op_arch = list(op.keys())[0]
            op_cfg = list(op.values())[0]
            op = create(op_arch, op_cfg, env_cfg)
            self.op_name2op[op_arch] = op

    def init_output(self, input):
        output = dict()
        for data_path in input:
            output[data_path] = {'pred': []}
        return output

    def update_output(self, input, output):
        for pred in input:
            image_path = pred['image_path']
            pred.pop("image_path")
            pred.pop("image_id") if "image_id" in pred.keys() else None
            output[image_path]['pred'].append(pred)
        return output

    def get_final_output(self, input, output, last_op=None):
        if last_op != 'PostProcess':
            input = self.update_output(input, output)
        for _, img_info in input.items():
            preds = img_info['pred']
            img_info['isNG'] = 0
            if any((pred['isNG'] == 1) for pred in preds):
                img_info['isNG'] = 1
        return input

    def run(self, input):
        output = self.init_output(input)
        # execute each operator according to toposort order
        for op_name, op in self.op_name2op.items():
            if op_name == 'PostProcess':
                input = self.update_output(input, output)
            input = op(input)

        return self.get_final_output(input, output, last_op=op_name)
