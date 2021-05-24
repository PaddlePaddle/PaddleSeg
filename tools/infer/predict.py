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
import time
import sys

import pynvml
import psutil
import GPUtil
import yaml
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import paddleseg.transforms as T
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger
from paddleseg.utils.visualize import get_pseudo_color_map
from benchmark_utils import PaddleInferBenchmark


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
        '--data_dir',
        dest='data_dir',
        help='The directory or path of the image to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--file_path',
        dest='file_path',
        help='The directory or path of the image to be predicted.',
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
        '--without_argmax',
        dest='without_argmax',
        help='Do not perform argmax operation on the predict result.',
        action='store_true')

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # params for prediction engine
    parser.add_argument("--device", type=str, default='gpu')
    parser.add_argument("--cpu_threads", type=int, default=6)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument(
        '--use_trt',
        dest='use_trt',
        help='Whether to use Nvidia TensorRT to accelerate prediction.',
        type=str2bool,
        default=False)
    parser.add_argument(
        '--precision',
        help='precision when using TensorRT prediction.',
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'int8'])
    return parser.parse_args()


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
        # pred_cfg.disable_glog_info()
        if self.args.device == 'gpu':
            pred_cfg.enable_use_gpu(100, 0)

            if self.args.use_trt:
                if args.precision == "fp32":
                    ptype = PrecisionType.Float32
                elif args.precision == "fp16":
                    ptype = PrecisionType.Half
                elif args.precision == "int8":
                    ptype = PrecisionType.Int8
                else:
                    raise ValueError(
                        "argument precision set wrong, it can only be fp32, fp16 and int8"
                    )

                pred_cfg.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=1,
                    min_subgraph_size=3,
                    precision_mode=ptype,
                    use_static=False,
                    use_calib_mode=False)
                min_input_shape = {
                    "x": [1, 3, 100, 100],
                    "bilinear_interp_v2_0.tmp_0": [1, 16, 25, 25],
                    "relu_10.tmp_0": [1, 16, 25, 25],
                    "relu_25.tmp_0": [1, 32, 13, 13],
                    "batch_norm_37.tmp_2": [1, 64, 7, 7],
                    "bilinear_interp_v2_1.tmp_0": [1, 16, 25, 25],
                    "bilinear_interp_v2_2.tmp_0": [1, 16, 25, 25],
                    "bilinear_interp_v2_3.tmp_0": [1, 32, 13, 13],
                    "relu_21.tmp_0": [1, 16, 25, 25],
                    "relu_29.tmp_0": [1, 64, 7, 7],
                    "relu_32.tmp_0": [1, 16, 13, 13],
                    "tmp_15": [1, 32, 13, 13]
                }
                max_input_shape = {
                    "x": [1, 3, 2000, 2000],
                    "bilinear_interp_v2_0.tmp_0": [1, 16, 500, 500],
                    "relu_10.tmp_0": [1, 16, 500, 500],
                    "relu_25.tmp_0": [1, 32, 250, 250],
                    "batch_norm_37.tmp_2": [1, 64, 125, 125],
                    "bilinear_interp_v2_1.tmp_0": [1, 16, 500, 500],
                    "bilinear_interp_v2_2.tmp_0": [1, 16, 500, 500],
                    "bilinear_interp_v2_3.tmp_0": [1, 32, 250, 250],
                    "relu_21.tmp_0": [1, 16, 500, 500],
                    "relu_29.tmp_0": [1, 64, 125, 125],
                    "relu_32.tmp_0": [1, 16, 250, 250],
                    "tmp_15": [1, 32, 250, 250]
                }
                opt_input_shape = {
                    "x": [1, 3, 192, 192],
                    "bilinear_interp_v2_0.tmp_0": [1, 16, 48, 48],
                    "relu_10.tmp_0": [1, 16, 48, 48],
                    "relu_25.tmp_0": [1, 32, 24, 24],
                    "batch_norm_37.tmp_2": [1, 64, 12, 12],
                    "bilinear_interp_v2_1.tmp_0": [1, 16, 48, 48],
                    "bilinear_interp_v2_2.tmp_0": [1, 16, 48, 48],
                    "bilinear_interp_v2_3.tmp_0": [1, 32, 24, 24],
                    "relu_21.tmp_0": [1, 16, 48, 48],
                    "relu_29.tmp_0": [1, 64, 12, 12],
                    "relu_32.tmp_0": [1, 16, 24, 24],
                    "tmp_15": [1, 32, 24, 24]
                }
                pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)
        else:
            pred_cfg.disable_gpu()
            pred_cfg.set_cpu_math_library_num_threads(args.cpu_threads)
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                pred_cfg.set_mkldnn_cache_capacity(10)
                pred_cfg.enable_mkldnn()

        # enable memory optim
        pred_cfg.enable_memory_optim()

        self.predictor = create_predictor(pred_cfg)
        self.config = pred_cfg

    def preprocess(self, img):
        return self.cfg.transforms(img)[0]

    def run(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        self.num = len(imgs)
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        results = []

        self.preprocess_time = Times()
        self.inference_time = Times()
        self.postprocess_time = Times()
        cpu_mem, gpu_mem = 0, 0
        gpu_id = 0
        gpu_util = 0

        iter_ = 0

        for i in range(0, self.num, self.args.batch_size):
            self.preprocess_time.start()
            data = np.array([
                self.preprocess(img) for img in imgs[i:i + self.args.batch_size]
            ])
            self.preprocess_time.end()

            # inference
            self.inference_time.start()
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            output_handle = self.predictor.get_output_handle(output_names[0])
            results.append(output_handle.copy_to_cpu())
            self.inference_time.end()

            gpu_util += get_current_gputil(gpu_id)
            cm, gm = get_current_memory_mb(gpu_id)
            cpu_mem += cm
            gpu_mem += gm

            iter_ += 1

        self.postprocess_time.start()
        self.postprocess(results, imgs)
        self.postprocess_time.end()
        self.avg_preprocess = self.preprocess_time.value() / self.num
        self.avg_inference = self.inference_time.value() / self.num
        self.avg_postprocess = self.postprocess_time.value() / self.num
        self.avg_cpu_mem = cpu_mem / iter_
        self.avg_gpu_mem = gpu_mem / iter_
        self.avg_gpu_util = gpu_util / iter_

    def postprocess(self, results, imgs):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        results = np.concatenate(results, axis=0)
        for i in range(results.shape[0]):
            if self.args.without_argmax:
                result = results[i]
            else:
                result = np.argmax(results[i], axis=0)
            result = get_pseudo_color_map(result)
            basename = os.path.basename(imgs[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))

    def report(self):
        perf_info = {
            'inference_time_s': self.avg_inference,
            'preprocess_time_s': self.avg_preprocess,
            'postprocess_time_s': self.avg_postprocess
        }
        model_info = {
            'model_name': self.args.cfg.split('/')[-2],
            'precision': self.args.precision
        }
        data_info = {
            'batch_size': self.args.batch_size,
            'shape': "dynamic_shape",
            'data_num': self.num
        }
        resource_info = {
            'cpu_rss_mb': self.avg_cpu_mem,
            'gpu_rss_mb': self.avg_gpu_mem,
            'gpu_util': self.avg_gpu_util
        }
        seg_log = PaddleInferBenchmark(self.config, model_info, data_info,
                                       perf_info, resource_info)
        seg_log('Seg')


# create time count class
class Times(object):
    def __init__(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += self.et - self.st
        else:
            self.time = self.et - self.st

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


def get_current_memory_mb(gpu_id=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    if gpu_id is not None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return cpu_mem, gpu_mem


def get_current_gputil(gpu_id):
    GPUs = GPUtil.getGPUs()
    gpu_load = GPUs[gpu_id].load
    return gpu_load


def get_images(file_path, data_dir):
    img_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            img_path, label = line.split()
            img_path = os.path.join(data_dir, img_path)
            img_list.append(img_path)
    return img_list


def main(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    print(info)

    predictor = Predictor(args)
    predictor.run(get_images(args.file_path, args.data_dir))
    predictor.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
