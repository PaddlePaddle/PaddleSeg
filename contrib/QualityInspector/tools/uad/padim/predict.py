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
import random
import argparse
from random import sample
from collections import OrderedDict

import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries

import paddle
import paddle.nn.functional as F

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)

import qinspector.uad.datasets.mvtec as mvtec
from qinspector.uad.models.padim import ResNet_PaDiM
from qinspector.cvlib.uad_configs import *


def argsparser():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="category name for MvTec AD dataset")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument('--resize', type=list or tuple, default=None)
    parser.add_argument('--crop_size', type=list or tuple, default=None)

    parser.add_argument("--save_pic", type=bool, default=True)
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
    model = ResNet_PaDiM(arch=args.backbone, pretrained=False)
    state = paddle.load(args.model_path)
    model.model.set_dict(state["params"])
    model.distribution = state["distribution"]
    model.eval()

    t_d, d = 448, 100  # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    print("Testing model for {} with sigle picture".format(class_name))

    # build datasets
    MVTecDataset = mvtec.MVTecDataset(is_predict=True)
    transform_x = MVTecDataset.get_transform_x()
    x = Image.open(args.img_path).convert('RGB')
    x = transform_x(x).unsqueeze(0)
    idx = paddle.to_tensor(sample(range(0, t_d), d))
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_imgs = []
    test_imgs.extend(x.cpu().detach().numpy())

    # model prediction
    with paddle.no_grad():
        outputs = model(x)
    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())
    for k, v in test_outputs.items():
        test_outputs[k] = paddle.concat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        layer_embedding = test_outputs[layer_name]
        layer_embedding = F.interpolate(
            layer_embedding, size=embedding_vectors.shape[-2:], mode="nearest")
        embedding_vectors = paddle.concat((embedding_vectors, layer_embedding),
                                          1)

    # randomly select d dimension
    embedding_vectors = paddle.index_select(embedding_vectors, idx, 1)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.shape
    embedding = embedding_vectors.reshape((B, C, H * W))
    # calculate mahalanobis distances
    mean, covariance = paddle.to_tensor(model.distribution[
        0]), paddle.to_tensor(model.distribution[1])
    inv_covariance = paddle.linalg.inv(covariance.transpose((2, 0, 1)))

    delta = (embedding - mean).transpose((2, 0, 1))

    distances = (paddle.matmul(delta, inv_covariance) * delta).sum(2).transpose(
        (1, 0))
    distances = distances.reshape((B, H, W))
    distances = paddle.sqrt(distances)

    # upsample
    score_map = F.interpolate(
        distances.unsqueeze(1),
        size=x.shape[2:],
        mode='bilinear',
        align_corners=False).squeeze(1).numpy()
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    if args.save_pic:
        save_name = os.path.join(args.save_path, args.backbone, args.category)
        dir_name = os.path.dirname(save_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plot_fig(test_imgs, scores, args.threshold, save_name, class_name)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Predict :  Picture {}".format(args.img_path) + " done!")


def plot_fig(test_img, scores, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        if i < 1:  # save one result
            fig_img.savefig(
                os.path.join(save_dir, class_name + '_pre_{}'.format(i)),
                dpi=100)

        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
