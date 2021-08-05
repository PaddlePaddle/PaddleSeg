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


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(
            self.dic['Deploy']['transforms'])
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

        return T.Compose(transforms)


class Predictor:
    def __init__(self, args):
        self.cfg = DeployConfig(args.cfg)
        self.args = args

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        # pred_cfg.disable_glog_info() # turn on if you don't want log info
        pred_cfg.enable_memory_optim()

        if args.device == 'gpu':
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            pred_cfg.enable_use_gpu(100, 0)
            pred_cfg.switch_ir_optim(True)
            precision_map = {
                "fp16": PrecisionType.Half,
                "fp32": PrecisionType.Float32,
                "int8": PrecisionType.Int8
            }
            precision_mode = precision_map[args.precision]

            if args.use_trt:
                pred_cfg.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=1,
                    min_subgraph_size=50,
                    precision_mode=precision_mode,
                    use_static=False,
                    use_calib_mode=False)
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 2000]}
                opt_input_shape = {"x": [1, 3, 192, 192]}
                pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)
        else:
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            pred_cfg.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                pred_cfg.set_mkldnn_cache_capacity(10)
                pred_cfg.enable_mkldnn()
            pred_cfg.set_cpu_math_library_num_threads(args.cpu_threads)

        self.predictor = create_predictor(pred_cfg)

        if args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name="hrnet_w18_small_v1",
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=args.save_log_path,
                inference_config=pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def preprocess(self, img):
        return self.cfg.transforms(img)[0]

    def run(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        num = len(imgs)
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        for i in range(0, num, self.args.batch_size):
            if args.benchmark and i > 0:
                self.autolog.times.start()
            data = np.array([
                self.preprocess(img) for img in imgs[i:i + self.args.batch_size]
            ])

            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)
            if args.benchmark and i > 0:
                self.autolog.times.stamp()

            self.predictor.run()

            results = output_handle.copy_to_cpu()
            if args.benchmark and i > 0:
                self.autolog.times.stamp()

            results = self.postprocess(results)

            if args.benchmark and i > 0:
                self.autolog.times.end(stamp=True)
            self.save_imgs(results, imgs)

    def postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def save_imgs(self, results, imgs):
        for i in range(results.shape[0]):
            result = get_pseudo_color_map(results[i])
            basename = os.path.basename(imgs[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
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
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
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
        "--benchmark",
        type=eval,
        default=False,
        help=
        "Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./log_output/",
        help="The file path to save log.")

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')

    return parser.parse_args()


def main(args):
    predictor = Predictor(args)
    imgs_list, _ = get_image_list(args.image_path)
    predictor.run(imgs_list)
    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
