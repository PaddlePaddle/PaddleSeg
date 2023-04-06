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
import sys
import time
import random
import argparse
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import paddle
from paddle.io import DataLoader

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import qinspector.uad.datasets.mvtec as mvtec
from qinspector.uad.models.patchcore import get_model
from qinspector.uad.utils.utils import str2bool
from qinspector.cvlib.uad_configs import ConfigParser
from val import val

textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = [
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
    'toothbrush', 'transistor', 'zipper'
]
CLASS_NAMES = textures + objects


def argsparser():
    parser = argparse.ArgumentParser('PatchCore')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="category name for MvTec AD dataset")
    parser.add_argument('--resize', type=list or tuple, default=None)
    parser.add_argument('--crop_size', type=list or tuple, default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument(
        "--k", type=int, default=None, help="used feature channels")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=[
            'sample', 'h_sample', 'ortho', 'svd_ortho', 'gaussian', 'coreset'
        ],
        help="projection method, one of [sample, ortho, svd_ortho, gaussian, coreset]"
    )
    parser.add_argument("--do_eval", type=bool, default=None)
    parser.add_argument("--save_pic", type=str2bool, default=None)

    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument(
        "--inc", action='store_true', help="use incremental cov & mean")
    parser.add_argument('--eval_PRO', type=bool, default=True)
    parser.add_argument(
        '--eval_threthold_step',
        type=int,
        default=500,
        help="threthold_step when computing PRO Score")
    parser.add_argument('--einsum', action='store_true')
    parser.add_argument('--non_partial_AUC', action='store_true')
    return parser.parse_args()


@paddle.no_grad()
def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.device.set_device(args.device)

    # build model
    model = get_model(args.method)(arch=args.backbone,
                                   pretrained=True,
                                   k=args.k,
                                   method=args.method)
    model.init_projection()
    model.eval()

    result = []
    assert args.category in mvtec.CLASS_NAMES + ['all', 'textures', 'objects']
    if args.category == 'all':
        class_names = mvtec.CLASS_NAMES
    elif args.category == 'textures':
        class_names = mvtec.textures
    elif args.category == 'objects':
        class_names = mvtec.objects
    else:
        class_names = [args.category]
    csv_columns = ['category', 'Image_AUROC', 'Pixel_AUROC', 'PRO_score']
    csv_name = os.path.join(
        args.save_path + f"/{args.method}_{args.backbone}_{args.k}",
        '{}_seed{}.csv'.format(args.category, args.seed))
    for i, class_name in enumerate(class_names):
        print("Training model {}/{} for {}".format(
            i + 1, len(class_names), class_name))
        # build datasets
        train_dataset = mvtec.MVTecDataset(
            args.data_path,
            class_name=class_name,
            is_train=True,
            resize=args.resize,
            cropsize=args.crop_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        train(args, model, train_dataloader, class_name)
        if args.do_eval:
            test_dataset = mvtec.MVTecDataset(
                args.data_path,
                class_name=class_name,
                is_train=False,
                resize=args.resize,
                cropsize=args.crop_size)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.test_batch_size,
                num_workers=args.num_workers)
            result.append(
                [class_name, *val(args, model, test_dataloader, class_name)])
            if args.category in ['all', 'textures', 'objects']:
                pd.DataFrame(
                    result,
                    columns=csv_columns).set_index('category').to_csv(csv_name)
        model.reset_stats()
    if args.do_eval:
        result = pd.DataFrame(result, columns=csv_columns).set_index('category')
        if not args.eval_PRO: del result['PRO_score']
        if args.category in ['all', 'textures', 'objects']:
            result.loc['mean'] = result.mean(numeric_only=True)
        print(result)
        print("Evaluation result saved at{}:".format(csv_name))
        result.to_csv(csv_name)


@paddle.no_grad()
def train(args, model, train_dataloader, class_name):
    epoch_begin = time.time()
    # extract train set features
    if args.inc:
        c = model.k  # args.k
        h = w = args.crop_size // 4
        N = 0  # sample num
        for x in tqdm(train_dataloader,
                      '| feature extraction | train | %s |' % class_name):
            # model prediction
            out = model(x)
            out = model.project(out, True)  # hwbc
            model.compute_stats_incremental(out)
            N += x.shape[0]
        del out, x
    else:
        outs = []
        for x in tqdm(train_dataloader,
                      '| feature extraction | train | %s |' % class_name):
            # model prediction
            out = model(x)
            out = model.project(out)
            outs.append(out)
        del out, x
        outs = paddle.concat(outs, 0)

    if args.inc:
        model.compute_inv_incremental(N)
    else:
        if args.einsum:
            model.compute_stats_einsum(outs)
        else:
            model.compute_stats(outs)
        del outs

    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(t))
    if args.save_model:
        print("Saving model...")
        save_name = os.path.join(
            args.save_path + f"/{args.method}_{args.backbone}_{args.k}",
            '{}.pdparams'.format(class_name))
        dir_name = os.path.dirname(save_name)
        os.makedirs(dir_name, exist_ok=True)
        state_dict = {
            "params": model.model.state_dict(),
            "stats": model._buffers,
        }
        paddle.save(state_dict, save_name)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Save model in {}".format(str(save_name)))


if __name__ == '__main__':
    main()
