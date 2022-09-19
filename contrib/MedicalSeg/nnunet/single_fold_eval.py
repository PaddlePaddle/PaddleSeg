# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

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

import argparse
import os
import shutil
import paddle
import numpy as np
import pickle
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from medicalseg.cvlibs import Config
from medicalseg.utils import get_sys_env, logger, config_check, utils

from nnunet.utils import DynamicPredictor, save_segmentation_nifti_from_softmax, aggregate_scores, determine_postprocessing, resample_and_save, predict_next_stage


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def evaluate(model, eval_dataset, args):
    """
    Launch evalution.
    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        args(str, optional): configura args.
    """
    model.eval()
    predictor = DynamicPredictor(model)
    plans = model.plans
    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params'][
            'force_separate_z']
        interpolation_order = plans['segmentation_export_params'][
            'interpolation_order']
        interpolation_order_z = plans['segmentation_export_params'][
            'interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
    output_base = args.val_save_folder
    output_folder = os.path.join(output_base,
                                 'fold_{}'.format(eval_dataset.fold))
    validation_raw_folder = os.path.join(output_folder, 'validation_raw')

    os.makedirs(validation_raw_folder, exist_ok=True)
    if not eval_dataset.data_aug_params['do_mirror']:
        raise RuntimeError(
            "We did not train with mirroring so you cannot do inference with mirroring enabled"
        )
    mirror_axes = eval_dataset.data_aug_params['mirror_axes']

    if args.predict_next_stage:
        next_stage_output_folder = eval_dataset.folder_with_segs_from_prev_stage
        os.makedirs(next_stage_output_folder, exist_ok=True)

    print('start evaluating...')
    pred_gt_tuples = []
    gt_niftis_folder = os.path.join(eval_dataset.preprocessed_dir,
                                    "gt_segmentations")  # gt dir
    for k in eval_dataset.dataset_val.keys():
        with open(eval_dataset.dataset[k]['properties_file'], 'rb') as f:
            properties = pickle.load(f)
        fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
        pred_gt_tuples.append([
            os.path.join(validation_raw_folder, fname + '.nii.gz'),
            os.path.join(gt_niftis_folder, fname + '.nii.gz'),
        ])
        if os.path.exists(
                os.path.join(validation_raw_folder, fname + '.nii.gz')):
            print('{} already exists, skip.'.format(
                os.path.join(validation_raw_folder, fname + '.nii.gz')))
            continue
        data = np.load(eval_dataset.dataset[k]['data_file'])['data']
        print(k, data.shape)
        data[-1][data[-1] == -1] = 0
        data = data[:-1]

        if eval_dataset.stage == 1 and eval_dataset.cascade:
            seg_pre_path = os.path.join(
                eval_dataset.folder_with_segs_from_prev_stage,
                k + "_segFromPrevStage.npz")
            if not os.path.exists(seg_pre_path):
                raise UserWarning(
                    'cannot find stage 1 segmentation result for {}.'.format(
                        seg_pre_path))
            seg_from_prev_stage = np.load(seg_pre_path)['data'][None]
            data = np.concatenate(
                (data, to_one_hot(seg_from_prev_stage[0],
                                  range(1, eval_dataset.num_classes))))

        argmax_pred, softmax_pred = predictor.predict_3D(
            data,
            do_mirroring=args.do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=args.use_sliding_window,
            step_size=args.step_size,
            patch_size=eval_dataset.patch_size,
            regions_class_order=None,
            use_gaussian=args.use_gaussian,
            pad_border_mode='constant',
            pad_kwargs=None,
            verbose=args.verbose,
            mixed_precision=args.precision == 'fp16')

        softmax_pred = softmax_pred.transpose(
            [0] + [i + 1 for i in eval_dataset.transpose_backward])
        softmax_fname = os.path.join(validation_raw_folder, fname + '.npz')

        if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):
            np.save(
                os.path.join(validation_raw_folder, fname + ".npy"),
                softmax_pred)
            softmax_pred = os.path.join(validation_raw_folder, fname + ".npy")

        save_segmentation_nifti_from_softmax(
            softmax_pred,
            os.path.join(validation_raw_folder, fname + '.nii.gz'), properties,
            interpolation_order, None, None, None, softmax_fname, None,
            force_separate_z, interpolation_order_z)

    _ = aggregate_scores(
        pred_gt_tuples,
        labels=list(range(eval_dataset.num_classes)),
        json_output_file=os.path.join(validation_raw_folder, "summary.json"),
        json_name=" val tiled %s" % (str(args.use_sliding_window)),
        json_author="medicalseg",
        json_task='task',
        num_threads=4)

    determine_postprocessing(
        output_folder,
        gt_niftis_folder,
        'validation_raw',
        final_subf_name='validation_raw' + "_postprocessed",
        debug=False)

    gt_nifti_folder = os.path.join(output_base, "gt_niftis")

    if not os.path.exists(gt_nifti_folder):
        os.makedirs(gt_nifti_folder, exist_ok=True)
    print('copy gt from {} to {}.'.format(gt_niftis_folder, gt_nifti_folder))

    gt_files = []
    for f_name in os.listdir(gt_niftis_folder):
        if os.path.isfile(os.path.join(gt_niftis_folder,
                                       f_name)) and f_name.endswith('.nii.gz'):
            gt_files.append(os.path.join(gt_niftis_folder, f_name))

    for f in gt_files:
        success = False
        attempts = 0
        e = None
        while not success and attempts < 10:
            try:
                shutil.copy(f, gt_nifti_folder)
                success = True
            except OSError as e:
                attempts += 1
        if not success:
            print("Could not copy gt nifti file %s into folder %s" %
                  (f, gt_nifti_folder))
            if e is not None:
                raise e

    if args.predict_next_stage:
        predict_next_stage(
            model,
            plans,
            eval_dataset,
            eval_dataset.preprocessed_dir,
            os.path.join(eval_dataset.preprocessed_dir,
                         plans['data_identifier'] + "_stage%d" % 1),
            args.precision == 'fp16',
            num_threads=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default="saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model/model.pdparams"
    )

    parser.add_argument(
        '--predict_next_stage',
        action='store_true',
        default=False,
        help='whether predict stage 2 training data.')

    parser.add_argument(
        '--val_save_folder',
        dest='val_save_folder',
        help='The path to val data predicted result',
        type=str,
        default="val_save_folder")

    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )

    parser.add_argument(
        "--do_mirroring",
        dest='do_mirroring',
        type=bool,
        default=True,
        help="Whether use mirroring when inference.")

    parser.add_argument(
        "--use_sliding_window",
        dest='use_sliding_window',
        type=bool,
        default=True,
        help="Whether use sliding window when inference.")

    parser.add_argument(
        "--step_size",
        dest='step_size',
        type=float,
        default=0.5,
        help="step size when predict.")

    parser.add_argument(
        "--use_gaussian",
        dest='use_gaussian',
        type=bool,
        default=True,
        help="Whether use gaussian.")

    parser.add_argument(
        "--verbose",
        dest='verbose',
        type=bool,
        default=True,
        help="Whether print log.")

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)

    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    config_check(cfg, val_dataset=val_dataset)

    evaluate(model, val_dataset, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
