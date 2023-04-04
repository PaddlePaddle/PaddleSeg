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
import datetime
from random import sample
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import qinspector.uad.datasets.mvtec as mvtec
from qinspector.uad.models.padim import ResNet_PaDiM
from qinspector.cvlib.uad_configs import *

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
    parser.add_argument("--do_val", type=bool, default=None)
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

    result = []
    assert args.category in mvtec.CLASS_NAMES
    csv_columns = ['category', 'Image_AUROC', 'Pixel_AUROC']
    csv_name = os.path.join(args.save_path + args.backbone,
                            '{}_seed{}.csv'.format(args.category, args.seed))
    # build model
    model = ResNet_PaDiM(arch=args.backbone, pretrained=False)
    state = paddle.load(args.model_path)
    model.model.set_dict(state["params"])
    model.distribution = state["distribution"]
    model.eval()

    t_d, d = fins[
        args.
        backbone], 100  # "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4}
    idx = paddle.to_tensor(sample(range(0, t_d), d))
    class_name = args.category
    print("Eval model for {}".format(class_name))

    # build datasets
    test_dataset = mvtec.MVTecDataset(
        args.data_path,
        class_name=class_name,
        is_train=False,
        resize=args.resize,
        cropsize=args.crop_size)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    result.append(
        [class_name, *val(args, model, test_dataloader, class_name, idx)])
    result = pd.DataFrame(result, columns=csv_columns).set_index('category')
    print(result)
    print("Evaluation result saved at{}:".format(csv_name))
    result.to_csv(csv_name)


def val(args, model, test_dataloader, class_name, idx):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Starting eval model...")
    total_roc_auc = []
    total_pixel_roc_auc = []

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader,
                             '| feature extraction | test | %s |' % class_name):

        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with paddle.no_grad():
            outputs = model(x)
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Eval model...")
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
    if paddle.device.is_compiled_with_cuda():

        def mahalanobis_pd(sample, mean, conv_inv):
            return paddle.sqrt(
                paddle.matmul(
                    paddle.matmul((sample - mean).t(), conv_inv), (sample - mean
                                                                   )))[0]

        B, C, H, W = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape((B, C, H * W)).cuda()
        model.distribution[0] = paddle.to_tensor(model.distribution[0]).cuda()
        model.distribution[1] = paddle.to_tensor(model.distribution[1]).cuda()
        dist_list = []
        for i in range(H * W):
            mean = model.distribution[0][:, i]
            conv_inv = paddle.linalg.inv(model.distribution[1][:, :, i])
            dist = [
                mahalanobis_pd(sample[:, i], mean, conv_inv).numpy()[0]
                for sample in embedding_vectors
            ]
            dist_list.append(dist)
    else:
        # calculate distance matrix
        B, C, H, W = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape((B, C, H * W)).numpy()
        dist_list = []
        for i in range(H * W):
            mean = model.distribution[0][:, i]
            conv_inv = np.linalg.inv(model.distribution[1][:, :, i])
            dist = [
                mahalanobis(sample[:, i], mean, conv_inv)
                for sample in embedding_vectors
            ]
            dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = paddle.to_tensor(dist_list)
    score_map = F.interpolate(
        dist_list.unsqueeze(1),
        size=x.shape[2:],
        mode='bilinear',
        align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)

    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=np.int64)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(),
                                                           scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)

    if args.save_pic:
        save_name = os.path.join(args.save_path, args.backbone, args.category)
        dir_name = os.path.dirname(save_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_name,
                 class_name)
    return np.mean(total_roc_auc), np.mean(total_pixel_roc_auc)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
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
                os.path.join(save_dir, class_name + '_val_{}'.format(i)),
                dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
