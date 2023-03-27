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
from random import sample
from collections import OrderedDict

import numpy as np
import pandas as pd

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import qinspector.uad.datasets.mvtec as mvtec
from qinspector.uad.models.padim import ResNet_PaDiM
from qinspector.cvlib.uad_configs import ConfigParser
from val import val


textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = [
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
    'toothbrush', 'transistor', 'zipper'
]
CLASS_NAMES = textures + objects
fins = {"resnet18": 448, "resnet50": 1792, "wide_resnet50_2": 1792}

def argsparser():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument('--resize', type=list or tuple, default=None)
    parser.add_argument('--crop_size', type=list or tuple, default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--do_eval", type=bool, default=None)
    parser.add_argument("--save_pic", type=bool, default=None)

    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--print_freq", type=int, default=20)
    return parser.parse_args()


def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.device.set_device(args.device)

    # build model
    model = ResNet_PaDiM(arch=args.backbone, pretrained=True)
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

    csv_columns = ['category', 'Image_AUROC', 'Pixel_AUROC']
    csv_name = os.path.join(args.save_path + args.backbone,
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
        t_d, d = fins[
            args.
            backbone], 100  # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4}
        idx = paddle.to_tensor(sample(range(0, t_d), d))
        train(args, model, train_dataloader, idx, class_name)
        if args.do_eval:
            test_dataset = mvtec.MVTecDataset(
                args.data_path,
                class_name=class_name,
                is_train=False,
                resize=args.resize,
                cropsize=args.crop_size)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers)

            result.append([
                class_name, *val(args, model, test_dataloader, class_name, idx)
            ])
            if args.category in ['all', 'textures', 'objects']:
                pd.DataFrame(
                    result,
                    columns=csv_columns).set_index('category').to_csv(csv_name)
    if args.do_eval:
        result = pd.DataFrame(result, columns=csv_columns).set_index('category')
        if args.category in ['all', 'textures', 'objects']:
            result.loc['mean'] = result.mean(numeric_only=True)
        print(result)
        print("Evaluation result saved at{}:".format(csv_name))
        result.to_csv(csv_name)


def train(args, model, train_dataloader, idx, class_name):
    train_outputs = OrderedDict(
        [('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    epoch_begin = time.time()
    end_time = time.time()

    for index, x in enumerate(train_dataloader):
        start_time = time.time()
        data_time = start_time - end_time

        # model prediction
        with paddle.no_grad():
            outputs = model(x)

        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())

        end_time = time.time()
        batch_time = end_time - start_time

        if index % args.print_freq == 0:
            print(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
                "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".
                format(0, index + 1,
                       len(train_dataloader), 0,
                       float(0), float(batch_time), float(data_time)))

    for k, v in train_outputs.items():
        train_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = train_outputs[layer_name]
        layer_embedding = F.interpolate(
            layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding),
                                          1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors, idx, 1)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape((B, C, H * W))
    mean = paddle.mean(embedding_vectors, axis=0).numpy()
    cov = paddle.zeros((C, C, H * W)).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(),
                              rowvar=False) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]
    model.distribution = train_outputs
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(0, t))
    if args.save_model:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Saving model...")
        save_name = os.path.join(args.save_path, args.backbone, args.category,
                                 '{}.pdparams'.format(class_name))
        dir_name = os.path.dirname(save_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        state_dict = {
            "params": model.model.state_dict(),
            "distribution": model.distribution,
        }
        paddle.save(state_dict, save_name)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Save model in {}".format(str(save_name)))


if __name__ == '__main__':
    main()
