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

import argparse
import time
import os
import sys
import cv2
import numpy as np
import paddle
import paddleseg.transforms as T
from paddleseg.cvlibs import Config
from paddleseg.core.infer import reverse_transform
from paddleseg.utils.visualize import get_pseudo_color_map
from paddleseg.utils import metrics
from paddleseg.cvlibs import SegBuilder

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig


def _transforms(dataset):
    transforms = []
    if dataset == "human":
        transforms.append(T.PaddingByAspectRatio(aspect_ratio=1.77777778))
        transforms.append(T.Resize(target_size=[398, 224]))
        transforms.append(T.Normalize())
    elif dataset == "cityscapes" or dataset == "ade":
        transforms.append(T.Normalize())
    return transforms


def find_images_with_bounding_size(eval_dataset: paddle.io.Dataset):
    max_length_index = -1
    max_width_index = -1
    min_length_index = -1
    min_width_index = -1

    max_length = float('-inf')
    max_width = float('-inf')
    min_length = float('inf')
    min_width = float('inf')
    for idx, data in enumerate(eval_dataset):
        image = np.array(data['img'])
        h, w = image.shape[-2:]
        if h > max_length:
            max_length = h
            max_length_index = idx
        if w > max_width:
            max_width = w
            max_width_index = idx
        if h < min_length:
            min_length = h
            min_length_index = idx
        if w < min_width:
            min_width = w
            min_width_index = idx
    print(f"Found max image length: {max_length}, index: {max_length_index}")
    print(f"Found max image width: {max_width}, index: {max_width_index}")
    print(f"Found min image length: {min_length}, index: {min_length_index}")
    print(f"Found min image width: {min_width}, index: {min_width_index}")
    return paddle.io.Subset(eval_dataset, [max_width_index, max_length_index,
                                           min_width_index, min_length_index])


def load_predictor(args):
    """
    load predictor func
    """
    rerun_flag = False
    model_file = os.path.join(args.model_path, args.model_filename)
    params_file = os.path.join(args.model_path, args.params_filename)
    pred_cfg = PredictConfig(model_file, params_file)
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)
    if args.device == "GPU":
        pred_cfg.enable_use_gpu(100, 0)
    else:
        pred_cfg.disable_gpu()
        pred_cfg.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.use_mkldnn:
            pred_cfg.enable_mkldnn()
            if args.precision == "int8":
                # Please ensure that the quantized ops during inference are the same as
                # the ops set in the qat training configuration file
                pred_cfg.enable_mkldnn_int8({"conv2d", "depthwise_conv2d"})

    if args.use_trt:
        # To collect the dynamic shapes of inputs for TensorRT engine
        dynamic_shape_file = os.path.join(args.model_path, "dynamic_shape.txt")
        if os.path.exists(dynamic_shape_file):
            pred_cfg.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                         True)
            # pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
            print("trt set dynamic shape done!")
            precision_map = {
                "fp16": PrecisionType.Half,
                "fp32": PrecisionType.Float32,
                "int8": PrecisionType.Int8
            }
            pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=4,
                precision_mode=precision_map[args.precision],
                use_static=True,
                use_calib_mode=False, )
        else:
            pred_cfg.disable_gpu()
            pred_cfg.set_cpu_math_library_num_threads(10)
            pred_cfg.collect_shape_range_info(dynamic_shape_file)
            print("Start collect dynamic shape...")
            rerun_flag = True

    pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
    # pred_cfg.delete_pass("gpu_cpu_map_matmul_v2_to_mul_pass")
    # pred_cfg.delete_pass("delete_quant_dequant_linear_op_pass")
    # pred_cfg.delete_pass("delete_weight_dequant_linear_op_pass")
    predictor = create_predictor(pred_cfg)
    return predictor, rerun_flag


def predict_image(args):
    """
    predict image func
    """
    transforms = _transforms(args.dataset)
    transform = T.Compose(transforms)

    # Step1: Load image and preprocess
    data = transform({'img': args.image_file})
    data = data['img'][np.newaxis, :]

    # Step2: Prepare predictor
    predictor, rerun_flag = load_predictor(args)

    # Step3: Inference
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    input_handle.reshape(data.shape)
    input_handle.copy_from_cpu(data)

    warmup, repeats = 0, 1
    if args.benchmark:
        warmup, repeats = 20, 100

    for i in range(warmup):
        predictor.run()

    start_time = time.time()
    for i in range(repeats):
        predictor.run()
        results = output_handle.copy_to_cpu()
        if rerun_flag:
            print(
                "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
            )
            return
    total_time = time.time() - start_time
    avg_time = float(total_time) / repeats
    print(
        f"[Benchmark]Average inference time: \033[91m{round(avg_time*1000, 2)}ms\033[0m"
    )

    # Step4: Post process
    if args.dataset == "human":
        results = reverse_transform(
            paddle.to_tensor(results), im.shape, transforms, mode="bilinear")
        results = np.argmax(results, axis=1)
    result = get_pseudo_color_map(results[0])

    # Step5: Save result to file
    if args.save_file is not None:
        result.save(args.save_file)
        print(f"Saved result to \033[91m{args.save_file}\033[0m")


def eval(args):
    """
    eval mIoU func
    """
    # DataLoader need run on cpu
    paddle.set_device("cpu")

    data_cfg = Config(args.config)
    builder = SegBuilder(data_cfg)

    eval_dataset = builder.val_dataset

    predictor, rerun_flag = load_predictor(args)

    if rerun_flag and args.use_multi_img_for_dynamic_shape_collect:
        print(
            "***** Try to find the images with the largest and smallest length and width respectively in the ADE20K "
            "dataset for collecting dynamic shape. *****"
        )
        eval_dataset = find_images_with_bounding_size(eval_dataset)

    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True)

    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    total_samples = len(eval_dataset)
    sample_nums = len(loader)
    batch_size = int(total_samples / sample_nums)
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    print("Start evaluating (total_samples: {}, total_iters: {}).".format(
        total_samples, sample_nums))
    for batch_id, data in enumerate(loader):
        image = np.array(data['img'])
        label = np.array(data['label']).astype("int64")
        ori_shape = np.array(label).shape[-2:]
        input_handle.reshape(image.shape)
        input_handle.copy_from_cpu(image)
        start_time = time.time()
        predictor.run()
        results = output_handle.copy_to_cpu()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        if rerun_flag:
            if args.use_multi_img_for_dynamic_shape_collect:
                if batch_id == sample_nums - 1:
                    print(
                        "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
                    )
                    return
                else:
                    continue
            else:
                print(
                    "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
                )
                return
        logit = reverse_transform(
            paddle.to_tensor(results).unsqueeze(0), data['trans_info'], mode="bilinear")
        pred = paddle.to_tensor(logit).squeeze(0)
        if len(
                pred.shape
        ) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype="int32")

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            paddle.to_tensor(label),
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()

    _, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                               label_area_all)
    _, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    _, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)

    time_avg = predict_time / sample_nums
    print(
        "[Benchmark]Batch size: {}, Inference time(ms): min={}, max={}, avg={}".
        format(batch_size,
               round(time_min * 1000, 2),
               round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    infor = "[Benchmark] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        total_samples, miou, acc, kappa, mdice)
    print(infor)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="inference model filepath")
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="model.pdiparams",
        help="params file name")
    parser.add_argument(
        "--image_file",
        type=str,
        default=None,
        help="Image path to be processed.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The path to save the processed image.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="human",
        choices=["human", "cityscapes", 'ade'],
        help="The type of given image which can be 'human' or 'cityscapes'.", )
    parser.add_argument(
        "--config", type=str, default=None, help="path to config.")
    parser.add_argument(
        "--benchmark",
        type=bool,
        default=False,
        help="Whether to run benchmark or not.")
    parser.add_argument(
        "--use_trt",
        type=bool,
        default=False,
        help="Whether to use tensorrt engine or not.")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="Choose the device you want to run, it can be: CPU/GPU, default is GPU",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument(
        "--use_multi_img_for_dynamic_shape_collect", 
        type=bool, 
        default=False, 
        help="Whether it is necessary to use multiple images to collect shape infomation,\
        When the image sizes in the data set are different, it needs to be set to True.")
    args = parser.parse_args()
    if args.image_file:
        predict_image(args)
    else:
        eval(args)
