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

import cv2
import tqdm
import yaml
import numpy as np
import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger

from ppmatting.utils import get_image_list, mkdir, estimate_foreground_ml
import ppmatting.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy for matting model')
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
        '--trimap_path',
        dest='trimap_path',
        help='The directory or path or file list of the triamp to help predicted.',
        type=str,
        default=None)
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
        '--fg_estimate',
        default=True,
        type=eval,
        choices=[True, False],
        help='Whether to estimate foreground when predicting.')

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
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)
        self._transforms = self.load_transforms(self.dic['Deploy'][
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

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))
        return T.Compose(transforms)


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str]): the path for images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args)

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
        data = np.array([cfg.transforms(imgs[i])[0]])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except:
            logger.info(
                "Auto tune fail. Usually, the error is out of GPU memory, "
                "because the model and image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


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

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        self.predictor = create_predictor(self.pred_cfg)

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = None if args.device == 'cpu' else 0
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id,
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
        logger.info("Using CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Using MKLDNN")
            # cache 1- different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("using GPU")
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
                min_subgraph_size=300,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args.auto_tuned_shape_file):
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

    def run(self, imgs, trimaps=None, imgs_dir=None):
        self.imgs_dir = imgs_dir
        num = len(imgs)
        input_names = self.predictor.get_input_names()
        input_handle = {}

        for i in range(len(input_names)):
            input_handle[input_names[i]] = self.predictor.get_input_handle(
                input_names[i])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        args = self.args

        for i in tqdm.tqdm(range(0, num, args.batch_size)):
            # warm up
            if i == 0 and args.benchmark:
                for _ in range(5):
                    img_inputs = []
                    if trimaps is not None:
                        trimap_inputs = []
                    trans_info = []
                    for j in range(i, i + args.batch_size):
                        img = imgs[i]
                        trimap = trimaps[i] if trimaps is not None else None
                        data = self._preprocess(img=img, trimap=trimap)
                        img_inputs.append(data['img'])
                        if trimaps is not None:
                            trimap_inputs.append(data['trimap'][
                                np.newaxis, :, :])
                        trans_info.append(data['trans_info'])
                    img_inputs = np.array(img_inputs)
                    if trimaps is not None:
                        trimap_inputs = (
                            np.array(trimap_inputs)).astype('float32')

                    input_handle['img'].copy_from_cpu(img_inputs)
                    if trimaps is not None:
                        input_handle['trimap'].copy_from_cpu(trimap_inputs)
                    self.predictor.run()
                    results = output_handle.copy_to_cpu()

                    results = results.squeeze(1)
                    for j in range(args.batch_size):
                        trimap = trimap_inputs[
                            j] if trimaps is not None else None
                        result = self._postprocess(
                            results[j], trans_info[j], trimap=trimap)

            # inference
            if args.benchmark:
                self.autolog.times.start()

            img_inputs = []
            if trimaps is not None:
                trimap_inputs = []
            trans_info = []
            for j in range(i, i + args.batch_size):
                img = imgs[i]
                trimap = trimaps[i] if trimaps is not None else None
                data = self._preprocess(img=img, trimap=trimap)
                img_inputs.append(data['img'])
                if trimaps is not None:
                    trimap_inputs.append(data['trimap'][np.newaxis, :, :])
                trans_info.append(data['trans_info'])
            img_inputs = np.array(img_inputs)
            if trimaps is not None:
                trimap_inputs = (np.array(trimap_inputs)).astype('float32')

            input_handle['img'].copy_from_cpu(img_inputs)
            if trimaps is not None:
                input_handle['trimap'].copy_from_cpu(trimap_inputs)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            if args.benchmark:
                self.autolog.times.stamp()

            results = output_handle.copy_to_cpu()

            results = results.squeeze(1)
            for j in range(args.batch_size):
                trimap = trimap_inputs[j] if trimaps is not None else None
                result = self._postprocess(
                    results[j], trans_info[j], trimap=trimap)
                self._save_imgs(result, imgs[i + j])

            if args.benchmark:
                self.autolog.times.end(stamp=True)
        logger.info("Finish")

    def _preprocess(self, img, trimap=None):
        data = {}
        data['img'] = img
        if trimap is not None:
            data['trimap'] = trimap
            data['gt_fields'] = ['trimap']
        data = self.cfg.transforms(data)
        return data

    def _postprocess(self, alpha, trans_info, trimap=None):
        """recover pred to origin shape"""
        if trimap is not None:
            trimap = trimap.squeeze(0)
            alpha[trimap == 0] = 0
            alpha[trimap == 255] = 1
        for item in trans_info[::-1]:
            if item[0] == 'resize':
                h, w = item[1][0], item[1][1]
                alpha = cv2.resize(
                    alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            elif item[0] == 'padding':
                h, w = item[1][0], item[1][1]
                alpha = alpha[:, :, 0:h, 0:w]
            else:
                raise Exception("Unexpected info '{}' in im_info".format(item[
                    0]))
        return alpha

    def _save_imgs(self, alpha, img_path):
        ori_img = cv2.imread(img_path)
        alpha = (alpha * 255).astype('uint8')

        if self.imgs_dir is not None:
            img_path = img_path.replace(self.imgs_dir, '')
        else:
            img_path = os.path.basename(img_path)
        name, ext = os.path.splitext(img_path)
        if name[0] == '/' or name[0] == '\\':
            name = name[1:]

        alpha_save_path = os.path.join(args.save_dir, name + '_alpha.png')
        rgba_save_path = os.path.join(args.save_dir, name + '_rgba.png')

        # save alpha
        mkdir(alpha_save_path)
        cv2.imwrite(alpha_save_path, alpha)

        # save rgba image
        mkdir(rgba_save_path)
        if args.fg_estimate:
            fg = estimate_foreground_ml(ori_img / 255.0, alpha / 255.0) * 255
        else:
            fg = ori_img
        fg = fg.astype('uint8')
        alpha = alpha[:, :, np.newaxis]
        rgba = np.concatenate([fg, alpha], axis=-1)
        cv2.imwrite(rgba_save_path, rgba)


def main(args):
    imgs_list, imgs_dir = get_image_list(args.image_path)
    if args.trimap_path is None:
        trimaps_list = None
    else:
        trimaps_list, _ = get_image_list(args.trimap_path)

    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    predictor = Predictor(args)
    predictor.run(imgs=imgs_list, trimaps=trimaps_list, imgs_dir=imgs_dir)

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
