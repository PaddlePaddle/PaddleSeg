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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve

import paddle
from paddle.io import DataLoader

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import qinspector.uad.datasets.mvtec as mvtec
from qinspector.uad.models.patchcore import get_model
from qinspector.uad.utils.utils import *
from qinspector.cvlib.uad_configs import ConfigParser

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
    parser.add_argument('--model_path', type=str, default=None)
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
    parser.add_argument("--k", type=int, default=None, help="feature used")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="projection method, one of ['sample','h_sample', 'ortho', 'svd_ortho', 'gaussian']"
    )
    parser.add_argument("--save_pic", type=str2bool, default=None)

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--eval_PRO', type=bool, default=True)
    parser.add_argument('--non_partial_AUC', action='store_true')
    parser.add_argument(
        '--eval_threthold_step',
        type=int,
        default=500,
        help="threthold_step when computing PRO Score and non_partial_AUC")
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

    result = []
    assert args.category in mvtec.CLASS_NAMES + ['all', 'textures', 'objects']
    class_names = mvtec.CLASS_NAMES if args.category == 'all' else [
        args.category
    ]
    csv_columns = ['category', 'Image_AUROC', 'Pixel_AUROC', 'PRO_score']
    csv_name = os.path.join(
        args.save_path + f"/{args.method}_{args.backbone}_{args.k}",
        '{}_seed{}.csv'.format(args.category, args.seed))
    for i, class_name in enumerate(class_names):
        print("Eval model {}/{} for {}".format(i + 1,
                                               len(class_names), class_name))

        # build model
        model_path = args.model_path or args.save_path + f"/{args.method}_{args.arch}_{args.k}" + '/{}.pdparams'.format(
            class_name)
        model = get_model(args.method)(arch=args.backbone,
                                       pretrained=False,
                                       k=args.k,
                                       method=args.method)
        state = paddle.load(model_path)
        model.model.set_dict(state["params"])
        model.load(state["stats"])
        model.eval()

        # build datasets
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
    result = pd.DataFrame(result, columns=csv_columns).set_index('category')
    if not args.eval_PRO: result = result.drop(columns="PRO_score")
    if args.category in ['all', 'textures', 'objects']:
        result.loc['mean'] = result.mean(numeric_only=True)
    print(result)
    print("Evaluation result saved at{}:".format(csv_name))
    result.to_csv(csv_name)


@paddle.no_grad()
def val(args, model, test_dataloader, class_name):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Starting eval model...")

    gt_list = []
    gt_mask_list = []
    test_imgs = []
    score_map = []
    paddle.device.set_device("gpu")
    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader,
                             '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        out = model(x)
        out = model.project(out)
        score_map.append(model.generate_scores_map(out, x.shape[-2:]))
    del out
    score_map, image_score = list(zip(*score_map))
    score_map = np.concatenate(score_map, 0)
    image_score = np.concatenate(image_score, 0)

    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    score_map = (score_map - min_score) / (max_score - min_score)
    print(f"max_score:{max_score} min_score:{min_score}")
    # calculate image-level ROC AUC score
    gt_list = np.asarray(gt_list)
    # fpr, tpr, _ = roc_curve(gt_list, image_score)
    img_auroc = compute_roc_score(
        gt_list, image_score, args.eval_threthold_step, args.non_partial_AUC)
    # get optimal threshold
    precision, recall, thresholds = precision_recall_curve(gt_list, image_score)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    print(f"F1 image:{f1.max()} threshold:{max_score}")
    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list, dtype=np.int64).squeeze()
    # fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_auroc = compute_roc_score(
        gt_mask.flatten(),
        score_map.flatten(), args.eval_threthold_step, args.non_partial_AUC)
    # get optimal threshold
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(),
                                                           score_map.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    print(f"F1 pixel:{f1.max()} threshold:{max_score}")

    # calculate Per-Region-Overlap Score
    total_PRO = compute_pro_score(
        gt_mask, score_map, args.eval_threthold_step,
        args.non_partial_AUC) if args.eval_PRO else None

    print([class_name, img_auroc, per_pixel_auroc, total_PRO])
    if args.save_pic:
        save_dir = os.path.join(
            args.save_path + f"/{args.method}_{args.backbone}_{args.k}",
            class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, score_map, gt_mask_list, threshold, save_dir,
                 class_name)
    return img_auroc, per_pixel_auroc, total_PRO


def plot_roc(fpr, tpr, score, save_dir, class_name, tag='pixel'):
    plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {score:.3f}")
    plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
    plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{class_name}_{tag}_roc_curve.png")
    plt.close()


def plot_roc_all(fprs, tprs, scores, class_names, save_dir, tag='pixel'):
    plt.figure()
    for fpr, tpr, score, class_name in zip(fprs, tprs, scores, class_names):
        plt.plot(
            fpr,
            tpr,
            marker="o",
            color="k",
            label=f"{class_name} AUROC: {score:.3f}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_roc_curve.png")
    plt.close()


if __name__ == '__main__':
    main()
