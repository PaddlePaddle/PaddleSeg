# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp
import argparse
from ast import literal_eval

import numpy as np
from PIL import Image
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.utils import logger, get_image_list
from paddleseg.deploy.infer import DeployConfig

from paddlepanseg.cvlibs.info_dicts import build_info_dict
from paddlepanseg.transforms.functional import id2rgb


def parse_infer_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument(
        '--config',
        dest='cfg',
        help="Config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        help="Directory or path or file list of the images to be predicted.",
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        help="Mini batch size on each GPU (or on CPU).",
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        help="Directory to save the predicted results.",
        type=str,
        default="./output")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu', 'xpu', 'npu'],
        default='gpu',
        help="Select which device to perform inference with. Defaults to gpu.")
    parser.add_argument(
        '--use_trt',
        default=False,
        type=literal_eval,
        choices=[True, False],
        help="Whether or not to use NVIDIA TensorRT to accelerate prediction.")
    parser.add_argument(
        '--precision',
        default='fp32',
        type=str,
        choices=['fp32', 'fp16', 'int8'],
        help="TensorRT precision.")
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help="Minimum subgraph size in TensorRT optimization.")
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=literal_eval,
        choices=[True, False],
        help="Whether or not to enable tuned dynamic shape. We uses some images to collect "
        "the dynamic shape for trt sub graph, which avoids setting dynamic shape manually."
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default='auto_tune_tmp.pbtxt',
        help="Path of the temp file to save tuned dynamic shape.")

    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help="Number of threads to use when using CPU.")
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=literal_eval,
        choices=[True, False],
        help="Whether or not to enable the use of mkldnn when using CPU.")

    parser.add_argument(
        '--benchmark',
        type=literal_eval,
        default=False,
        choices=[True, False],
        help="Whether or not to print information about environment, model, configuration and performance."
    )
    parser.add_argument(
        '--model_name',
        default='',
        type=str,
        help="When `--benchmark` is True, the specified model name is displayed."
    )
    parser.add_argument(
        '--print_detail',
        default=True,
        type=literal_eval,
        choices=[True, False],
        help="Whether or not to print GLOG information of Paddle Inference.")

    return parser.parse_args(*args, **kwargs)


def use_auto_tune(args):
    return hasattr(PredictConfig, 'collect_shape_range_info') \
        and hasattr(PredictConfig, 'enable_tuned_tensorrt_dynamic_shape') \
        and args.device == 'gpu' and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.
    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support `auto_tune`, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = {'img': imgs[i]}
            data = np.array([cfg.transforms(data)['img']])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.error(str(e))
            logger.error(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if osp.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune succeded.\n")


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        For the usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        elif args.device == 'npu':
            self.pred_cfg.enable_custom_device('npu')
        elif args.device == 'xpu':
            self.pred_cfg.enable_xpu()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.error(str(e))
            logger.error(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set `--enable_auto_tune=True` to use the auto tuning function. \n"
            )
            exit(1)

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

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
            # Cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                osp.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i in range(0, len(imgs_path), args.batch_size):
            # Warm up
            if i == 0 and args.benchmark:
                for j in range(5):
                    data = np.array([
                        self._preprocess(img)
                        for img in imgs_path[0:args.batch_size]
                    ])
                    input_handle.reshape(data.shape)
                    input_handle.copy_from_cpu(data)
                    self.predictor.run()
                    results = output_handle.copy_to_cpu()
                    results = self._postprocess(results)

            # Do inference
            if args.benchmark:
                self.autolog.times.start()

            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + args.batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            results = output_handle.copy_to_cpu()
            if args.benchmark:
                self.autolog.times.stamp()

            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            self._save_imgs(results, imgs_path[i:i + args.batch_size])
        logger.info("Inference finished!")

    def _preprocess(self, img):
        data = build_info_dict(_type_='sample', img=img)
        return self.cfg.transforms(data)['img']

    def _postprocess(self, results):
        # Do nothing
        return results

    def _save_imgs(self, results, imgs_path):
        for i in range(results.shape[0]):
            result = id2rgb(results[i][0])
            result = Image.fromarray(result).convert('RGB')
            basename = osp.basename(imgs_path[i])
            basename, _ = osp.splitext(basename)
            basename = f'{basename}.png'
            result.save(osp.join(self.args.save_dir, basename))


def infer_with_args(args):
    imgs_list, _ = get_image_list(args.image_path)

    # Collect dynamic shape by `auto_tune`
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # Create and run predictor
    predictor = Predictor(args)
    predictor.run(imgs_list)

    if use_auto_tune(args) and \
        osp.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_infer_args()
    infer_with_args(args)
