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
import yaml
import numpy as np
import paddle
import paddle.nn.functional as F

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

from paddleseg.cvlibs import manager
from paddleseg.utils import logger, metrics, progbar

from infer import Predictor, DeployConfig, use_auto_tune


def parse_args():
    parser = argparse.ArgumentParser(description='Model Infer')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)

    parser.add_argument(
        '--dataset_type',
        help='The name of dataset, such as Cityscapes, PascalVOC and ADE20K.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--dataset_path',
        help='The directory of the dataset to be predicted. If set dataset_path, '
        'it use the test and label images to calculate the mIoU.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--dataset_mode',
        help='The dataset mode, such as train, val.',
        type=str,
        default="val")
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
        '--batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)

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

    parser.add_argument(
        '--with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        help='Print GLOG information of Paddle Inference.',
        action='store_true')

    return parser.parse_args()


def get_dataset(args):
    comp = manager.DATASETS
    if args.dataset_type not in comp.components_dict:
        raise RuntimeError("The dataset is not supported.")

    cfg = DeployConfig(args.cfg)

    if args.resize_width == 0 and args.resize_height == 0:
        transforms = cfg.transforms.transforms
    else:
        # load and add resize to transforms
        assert args.resize_width > 0 and args.resize_height > 0
        with codecs.open(args.cfg, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        transforms_dic = dic['Deploy']['transforms']
        transforms_dic.insert(
            0, {
                "type": "Resize",
                'target_size': [args.resize_width, args.resize_height]
            })
        transforms = DeployConfig.load_transforms(transforms_dic).transforms

    kwargs = {
        'transforms': transforms,
        'dataset_root': args.dataset_path,
        'mode': args.dataset_mode
    }
    dataset = comp[args.dataset_type](**kwargs)
    return dataset


def auto_tune(args, dataset, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        dataset(dataset): an dataset.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args)

    num = min(len(dataset), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for idx, (img, _) in enumerate(dataset):
        data = np.array([img])
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

        if idx + 1 >= num:
            break

    logger.info("Auto tune success.\n")


class DatasetPredictor(Predictor):
    def __init__(self, args):
        super().__init__(args)

    def run_dataset(self):
        """
        Read the data from dataset and calculate the accurary of the inference model.
        """
        dataset = get_dataset(self.args)

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        total_time = 0
        progbar_val = progbar.Progbar(target=len(dataset), verbose=1)

        for idx, (img, label) in enumerate(dataset):
            data = np.array([img])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            start_time = time.time()
            self.predictor.run()
            end_time = time.time()
            total_time += (end_time - start_time)

            pred = output_handle.copy_to_cpu()
            pred = self._postprocess(pred)
            pred = paddle.to_tensor(pred, dtype='int64')
            label = paddle.to_tensor(label, dtype="int32")
            if pred.shape != label.shape:
                label = paddle.unsqueeze(label, 0)
                label = F.interpolate(label, pred.shape[-2:])
                label = paddle.squeeze(label, 0)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                dataset.num_classes,
                ignore_index=dataset.ignore_index)

            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area

            progbar_val.update(idx + 1)

        class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                           label_area_all)
        class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
        kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

        logger.info(
            "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(
                len(dataset), miou, acc, kappa))
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Average time: %.3f ms/img" %
                    (total_time / len(dataset)) * 1000)


def main(args):
    if use_auto_tune(args):
        dataset = get_dataset(args)
        tune_img_nums = 10
        auto_tune(args, dataset, tune_img_nums)

    predictor = DatasetPredictor(args)
    predictor.run_dataset()

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)


if __name__ == '__main__':
    """
    Based on the infer config and dataset, this program read the test and
    label images, applys the transfors, run the predictor, ouput the accuracy.

    For example:
    python deploy/python/infer_dataset.py \
        --config path/to/bisenetv2/deploy.yaml \
        --dataset_type Cityscapes \
        --dataset_path path/to/cityscapes

    python deploy/python/infer_dataset.py \
        --config path/to/bisenetv2/deploy.yaml \
        --dataset_type Cityscapes \
        --dataset_path path/to/cityscapes \
        --device gpu \
        --use_trt True \
        --enable_auto_tune True
    """
    args = parse_args()
    main(args)
