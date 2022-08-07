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
import nibabel as nib
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
import paddle

import medicalseg.transforms as T
from medicalseg.cvlibs import manager
from medicalseg.utils import get_sys_env, logger, get_image_list
# from medicalseg.utils.visualize import get_pseudo_color_map
from medicalseg.core.infer import sliding_window_inference
# from tools import HUnorm, resample
# from tools import Prep
from skimage.transform import resize

def resize_image(image,new_shape,order=3,cval=0):
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    return resize(image,new_shape,order,cval=cval,**kwargs)

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

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

    parser.add_argument(
        '--use_swl',
        default=False,
        type=eval,
        help='use sliding_window_inference')

    parser.add_argument('--use_warmup', default=True, type=eval, help='warmup')

    parser.add_argument('--img_shape', default=128, type=int, help='img_shape')

    parser.add_argument('--is_nhwd', default=True, type=eval, help='is_nhwd')
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
            ctype = t.pop('type', None)
            if ctype is not None:
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


class ModelLikeInfer:
    def __init__(self, input_handle, output_handle, predictor):
        self.input_handle = input_handle
        self.output_handle = output_handle
        self.predictor = predictor

    def infer_likemodel(self, input_handle, output_handle, predictor, data):
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data.numpy())
        predictor.run()
        return paddle.to_tensor(output_handle.copy_to_cpu())

    def infer_model(self, data):
        return (self.infer_likemodel(self.input_handle, self.output_handle,
                                     self.predictor, data), )


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)
        self.target_spacing = [1.52, 1.52, 6.35]
        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        self.predictor = create_predictor(self.pred_cfg)

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
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
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

    def run(self, img_path):

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

            # inference
        if args.benchmark:
            self.autolog.times.start()
        data = self._preprocess(img_path)

        if args.benchmark:
            self.autolog.times.stamp()

        if args.use_swl:

            infer_like_model = ModelLikeInfer(input_handle, output_handle,
                                              self.predictor)
            data = paddle.to_tensor(data)
            # if args.is_nhwd:
            #     data = paddle.squeeze(data, axis=1)
            data=data.unsqueeze(0).unsqueeze(0)
            results = sliding_window_inference(
                data, (14, 160, 160), 1,
                infer_like_model.infer_model)

            results = results[0]

        else:
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            self.predictor.run()

            results = output_handle.copy_to_cpu()

        if args.benchmark:
            self.autolog.times.stamp()

        results = self._postprocess(results)

        if args.benchmark:
            self.autolog.times.end(stamp=True)

        # self._save_npy(results, imgs_path[i:i + args.batch_size])
        self.save_medical_label(results,img_path)
    logger.info("Finish")


    def _preprocess(self, img):

        data_array,original_spacing = self.load_medical_data(img)
        self.shape = data_array.shape
        new_shape = np.round(
            ((np.array(original_spacing) / np.array(self.target_spacing)).astype(float) * np.array(self.shape))).astype(
            int)
        data_array = resize_image(data_array, new_shape)
        # 将数据从hwd转化为dhw
        # import pdb
        # pdb.set_trace()
        data_array = np.transpose(data_array, [2, 0, 1])
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std > 0:
            data_array = (data_array - mean) / std
        else:
            data_array = (data_array - mean) / (std + 1e-8)


        return data_array.astype("float32")

    def _postprocess(self, result):
        "results is numpy array, optionally postprocess with argmax"
        if self.args.with_argmax:
            result= np.argmax(result, axis=1)

        result = np.transpose(result[0], [1, 2, 0])
        result = resize_segmentation(result, self.shape)
        return result

    def _save_npy(self, results, imgs_path):
        for i in range(results.shape[0]):
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.npy'
            np.save(os.path.join(self.args.save_dir, basename), results)
    def save_medical_label(self,result,img_path):
        basename = os.path.basename(img_path)
        nlabel = nib.Nifti1Image(result, self.nimg.affine, header=self.nimg.header)
        nib.save(nlabel, os.path.join(self.args.save_dir, basename))

    def load_medical_data(self,filename):
        self.nimg = nib.load(filename)
        data_array = self.nimg.get_data()
        original_spacing = self.nimg.header["pixdim"][1:4]
        return data_array,original_spacing

def main(args):
    imgs_list = get_image_list(
        args.image_path)  # get image list from image path

    # support autotune to collect dynamic shape, works only with trt on.
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # infer with paddle inference.
    predictor = Predictor(args)
    for filename in imgs_list:
        predictor.run(filename)

    if use_auto_tune(args) and \
            os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    # test the speed.
    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
