# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import codecs
import os
import sys
import cv2

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import numpy as np
import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, get_image_list

sys.path.append('../..')
from third_party import tusimple_processor
import paddleseg.transforms.transforms as T


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms, to_rgb=False)


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()
        self._init_cpu_config()

        self.predictor = create_predictor(self.pred_cfg)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def run(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_seg_handle = self.predictor.get_output_handle(output_names[0])

        args = self.args
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        cut_height = 160
        num_classes = 7
        postprocessor = tusimple_processor.TusimpleProcessor(
            num_classes=num_classes,
            cut_height=cut_height,
            save_dir=args.save_dir)

        for i, im_path in enumerate(imgs):
            im = cv2.imread(im_path)
            im = im[cut_height:, :, :]
            im = im.astype('float32')
            im, _ = self.cfg.transforms(im)
            im = im[np.newaxis, ...]

            input_handle.reshape(im.shape)
            input_handle.copy_from_cpu(im)

            self.predictor.run()

            seg_results = output_seg_handle.copy_to_cpu()

            # get lane points
            seg_results = paddle.to_tensor([seg_results])
            postprocessor.predict(seg_results, im_path)
        logger.info("Finish")


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')
    parser.add_argument(
        '--use_cpu',
        dest='use_cpu',
        help='Whether to use X86 CPU for inference. Uses GPU in default.',
        action='store_false')

    parser.add_argument(
        '--print_detail',
        dest='print_detail',
        help='Print GLOG information of Paddle Inference.',
        action='store_false')

    return parser.parse_args()


def main(args):
    imgs_list, _ = get_image_list(args.image_path)
    predictor = Predictor(args)
    predictor.run(imgs_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
