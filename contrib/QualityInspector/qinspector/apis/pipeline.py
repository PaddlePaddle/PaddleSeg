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

import glob
import json
import os
import os.path as osp

from qinspector.cvlib.configs import ConfigParser
from qinspector.cvlib.framework import Builder
from qinspector.utils.logger import setup_logger
from qinspector.utils.visualizer import show_result

logger = setup_logger('Pipeline')


class Pipeline(object):
    def __init__(self, cfg):
        config = ConfigParser(cfg)
        config.print_cfg()
        self.model_cfg, self.env_cfg = config.parse()
        self.modules = Builder(self.model_cfg, self.env_cfg)
        self.output_dir = self.env_cfg.get('output_dir', 'output')
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.visualize = self.env_cfg.get('visualize', False)
        self.save = self.env_cfg.get('save', False)

    def _parse_input(self, input):
        im_exts = ['jpg', 'jpeg', 'png', 'bmp']
        im_exts += [ext.upper() for ext in im_exts]

        if isinstance(input, (list, tuple)) and isinstance(input[0], str):
            input_type = "image"
            images = [
                image for image in input
                if any([image.endswith(ext) for ext in im_exts])
            ]
            assert len(images) > 0, "no image found"
            logger.info("Found {} inference images in total.".format(
                len(images)))
            return images, input_type

        if osp.isdir(input):
            input_type = "image"
            logger.info(
                'Input path is directory, search the images automatically')
            images = set()
            infer_dir = osp.abspath(input)
            for ext in im_exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            assert len(images) > 0, "no image found"
            logger.info("Found {} inference images in total.".format(
                len(images)))
            return images, input_type

        logger.info('Input path is {}'.format(input))
        input_ext = osp.splitext(input)[-1][1:]
        if input_ext in im_exts:
            input_type = "image"
            return [input], input_type
        raise ValueError("Unsupported input format: {}".format(input_ext))

    def predict_images(self, input):
        results = self.modules.run(input)

        if self.save:
            logger.info("Save prediction to {}".format(
                osp.join(self.output_dir, 'output.json')))
            with open(osp.join(self.output_dir, 'output.json'), "w") as f:
                json.dump(results, f, indent=2)

        if self.visualize:
            logger.info("visualize prediction to {}".format(
                osp.join(self.output_dir, 'show')))
            show_result(results, osp.join(self.output_dir, 'show'))
        return results

    def run(self, input):
        input, input_type = self._parse_input(input)
        if input_type == "image":
            results = self.predict_images(input)
        else:
            raise ValueError("Unexpected input type: {}".format(input_type))
        return results
