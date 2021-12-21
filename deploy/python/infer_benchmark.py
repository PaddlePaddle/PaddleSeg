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
import time

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleseg.transforms as T
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.utils.visualize import get_pseudo_color_map

from infer import use_auto_tune, auto_tune, DeployConfig, Predictor


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")
    parser.add_argument(
        '--resize_width',
        help='Set the resize width to acclerate the test. In default, it is 0, '
        'which means use the origin width.',
        type=int,
        default=0)
    parser.add_argument(
        '--resize_height',
        help='Set the resize height to acclerate the test. In default, it is 0, '
        'which means use the origin height.',
        type=int,
        default=0)

    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help=
        'Whether to enable tuned dynamic shape. We uses some images to collect '
        'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default="auto_tune_tmp.pbtxt",
        help='The temp file to save tuned dynamic shape.')

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

    parser.add_argument('--warmup', default=50, type=int, help='')
    parser.add_argument('--repeats', default=100, type=int, help='')

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        dest='print_detail',
        help='Print GLOG information of Paddle Inference.',
        action='store_true')

    return parser.parse_args()


class PredictorBenchmark(Predictor):
    def run(self, img_path):
        args = self.args
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        img_data = np.array([self._preprocess(img_path)])
        print(img_data.shape)
        input_handle.reshape(img_data.shape)
        input_handle.copy_from_cpu(img_data)

        logger.info("Warmup")
        for _ in range(args.warmup):
            self.predictor.run()

        logger.info("Infer")
        start_time = time.time()
        for _ in range(args.repeats):
            self.predictor.run()
        end_time = time.time()

        results = output_handle.copy_to_cpu()
        results = self._postprocess(results)

        self._save_imgs(results)

        avg_time = (end_time - start_time) * 1000 / args.repeats
        logger.info("Avg Time: {:.3}ms".format(avg_time))

    def _preprocess(self, img_path):
        if self.args.resize_width == 0 and self.args.resize_height == 0:
            return self.cfg.transforms(img_path)[0]
        else:
            assert args.resize_width > 0 and args.resize_height > 0
            with codecs.open(args.cfg, 'r', 'utf-8') as file:
                dic = yaml.load(file, Loader=yaml.FullLoader)
            transforms_dic = dic['Deploy']['transforms']
            transforms_dic.insert(
                0, {
                    "type": "Resize",
                    'target_size': [args.resize_width, args.resize_height]
                })
            transforms = DeployConfig.load_transforms(transforms_dic)
            return transforms(img_path)[0]

    def _save_imgs(self, results):
        for i in range(results.shape[0]):
            img = get_pseudo_color_map(results[i])
            basename = os.path.basename(self.args.image_path)
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            img.save(os.path.join(self.args.save_dir, basename))


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if use_auto_tune(args):
        auto_tune(args, args.image_path, 1)

    predictor = PredictorBenchmark(args)
    predictor.run(args.image_path)

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
