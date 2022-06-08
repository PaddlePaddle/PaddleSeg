# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import codecs
import warnings
import argparse

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import functools
import numpy as np

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import medicalseg.transforms as T
from medicalseg.cvlibs import manager
from medicalseg.utils import get_sys_env, logger, get_image_list
from medicalseg.utils.visualize import get_pseudo_color_map



from medicalseg.cvlibs import Config
from medicalseg.core import evaluate
from medicalseg.utils import get_sys_env, logger, config_check, utils


from tools import HUnorm, resample
from tools import Prep

from medicalseg.core.infer_window  import  sliding_window_inference
import paddle

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
        '--model_path',
        dest='model_path',
        help='The path of model ',
        type=str,
        default=None,
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
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to enable tuned dynamic shape. We uses some images to collect '
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

    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()




class ModelLike_infer:
    def __init__(self, input_handle,output_handle,predictor):
        self.input_handle  = input_handle
        self.output_handle = output_handle
        self.predictor = predictor


    def infer_likemodel(self,input_handle, output_handle, predictor,data):
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data.numpy())
        predictor.run()
        return paddle.to_tensor(output_handle.copy_to_cpu())

    def infer_model(self,data):

        return (self.infer_likemodel(self.input_handle, self.output_handle, self.predictor,data),)
           





class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        # self.cfg = DeployConfig(args.cfg)

        cfg = Config(args.cfg)
        model = cfg.model
        if args.model_path:
            utils.load_entire_model(model, args.model_path)
            logger.info('Loaded trained params of model successfully')
        self.model=model

        # self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        # self.predictor = create_predictor(self.pred_cfg)

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=None,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        "初始化基础配置"
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
      

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
       

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        # input_names = self.predictor.get_input_names()
        # input_handle = self.predictor.get_input_handle(input_names[0])
        # output_names = self.predictor.get_output_names()
        # output_handle = self.predictor.get_output_handle(output_names[0])
        results = []

        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # infer_likemodel = ModelLike_infer(input_handle,output_handle,self.predictor)

        for i in range(0, len(imgs_path), args.batch_size):
            # # warm up
            # if i == 0 and args.benchmark:
            #     for j in range(5):
            #         data = np.array([
            #             self._preprocess(img)  # load from original
            #             for img in imgs_path[0:args.batch_size]
            #         ])
            #         input_handle.reshape(data.shape)
            #         input_handle.copy_from_cpu(data)
            #         self.predictor.run()
            #         results = output_handle.copy_to_cpu()
            #         results = self._postprocess(results)
            # inference
            if args.benchmark:
                self.autolog.times.start()
            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + args.batch_size]
            ])


            data=paddle.to_tensor(data)

            if args.benchmark:
                self.autolog.times.stamp()



            results = sliding_window_inference(data,(128,128,128),1,self.model)
            results=paddle.to_tensor(results)           

            if args.benchmark:
                self.autolog.times.stamp()


            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            self._save_npy(results, imgs_path[i:i + args.batch_size])
        logger.info("Finish")

    def _preprocess(self, img):
        """load img and transform it

        Args:
        Img(str): A batch of image path

        """
        preT=T.Compose([])


        return preT(img)[0]

    def _postprocess(self, results):
        "results is numpy array, optionally postprocess with argmax"
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def _save_npy(self, results, imgs_path):
        for i in range(results.shape[0]):
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.npy'
            np.save(os.path.join(self.args.save_dir, basename), results)


def main(args):
    imgs_list = get_image_list(
        args.image_path)  # get image list from image path

    # # support autotune to collect dynamic shape, works only with trt on.
    # if use_auto_tune(args):
    #     tune_img_nums = 10
    #     auto_tune(args, imgs_list, tune_img_nums)

    # infer with paddle inference.
    predictor = Predictor(args)
    predictor.run(imgs_list)

    # if use_auto_tune(args) and \
    #     os.path.exists(args.auto_tuned_shape_file):
    #     os.remove(args.auto_tuned_shape_file)

    # test the speed.
    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
