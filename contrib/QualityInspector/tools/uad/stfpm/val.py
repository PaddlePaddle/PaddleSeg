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
import datetime
import argparse
from glob import glob

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.vision import transforms

from qinspector.uad.models.stfpm import ResNet_MS3
from qinspector.uad.datasets.mvtec import MVTecDatasetSTFPM, load_gt
from qinspector.uad.utils.utils import eval_metric
from qinspector.cvlib.uad_configs import ConfigParser


def argsparser():
    parser = argparse.ArgumentParser("STFPM")
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
    parser.add_argument('--resize', type=list, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default=None, help="student checkpoint")
    parser.add_argument("--compute_pro", type=bool, default=None)
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
    teacher = ResNet_MS3(arch=args.backbone, pretrained=True)
    student = ResNet_MS3(arch=args.backbone, pretrained=False)

    # build datasets
    transform = transforms.Compose([
        transforms.Resize(args.resize), transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_neg_image_list = sorted(
        glob(
            os.path.join(args.data_path, args.category, 'test', 'good',
                         '*.png')))
    test_pos_image_list = set(
        glob(
            os.path.join(args.data_path, args.category, 'test', '*',
                         '*.png'))) - set(test_neg_image_list)
    test_pos_image_list = sorted(list(test_pos_image_list))
    test_neg_dataset = MVTecDatasetSTFPM(
        test_neg_image_list, transform=transform)
    test_pos_dataset = MVTecDatasetSTFPM(
        test_pos_image_list, transform=transform)
    test_neg_loader = DataLoader(
        test_neg_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)
    test_pos_loader = DataLoader(
        test_pos_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)

    saved_dict = paddle.load(args.model_path)
    print('load ' + args.model_path)
    student.load_dict(saved_dict)

    val(args,
        student,
        teacher,
        test_pos_loader,
        test_neg_loader,
        epoch=None,
        eval_pro=args.compute_pro)


def cal_error(args, teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros(
        (len(loader.dataset), args.resize[0] // 4, args.resize[1] // 4))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        with paddle.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], axis=1)
            s_feat[j] = F.normalize(s_feat[j], axis=1)
            sm = paddle.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
            sm = F.interpolate(
                sm,
                size=(args.resize[0] // 4, args.resize[1] // 4),
                mode='bilinear',
                align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i:i + batch_img.shape[0]] = score_map.squeeze().cpu().numpy()
        i += batch_img.shape[0]
    return loss_map


def val(args,
        student,
        teacher,
        test_pos_loader,
        test_neg_loader,
        epoch=None,
        eval_pro=False):
    category = args.category
    gt = load_gt(args.data_path, category, args.resize)
    pos = cal_error(args, teacher, student, test_pos_loader)
    neg = cal_error(args, teacher, student, test_neg_loader)

    scores = []
    for i in range(len(pos)):
        temp = cv2.resize(pos[i], (args.resize[0], args.resize[1]))
        scores.append(temp)
    for i in range(len(neg)):
        temp = cv2.resize(neg[i], (args.resize[0], args.resize[1]))
        scores.append(temp)

    scores = np.stack(scores)
    neg_gt = np.zeros((len(neg), args.resize[0], args.resize[1]), dtype=np.bool)
    gt_pixel = np.concatenate((gt, neg_gt), 0)
    gt_image = np.concatenate(
        (np.ones(
            pos.shape[0], dtype=np.bool), np.zeros(
                neg.shape[0], dtype=np.bool)),
        0)

    if eval_pro:  ## very slow
        pro = eval_metric(gt_pixel, scores, metric='pro')
        auc_pixel = eval_metric(
            gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = eval_metric(
            gt_image, scores.max(-1).max(-1), metric='roc')
        if epoch != None:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
                  'Epoch: {}\tCatergory: {:s}\tPixel-AUC: {:.6f}'
                  '\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(
                      epoch, category, auc_pixel, auc_image_max, pro))
        else:
            print(
                'Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.
                format(category, auc_pixel, auc_image_max, pro))
    else:
        auc_pixel = eval_metric(
            gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = eval_metric(
            gt_image, scores.max(-1).max(-1), metric='roc')
        if epoch != None:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
                  'Epoch: {}\tCatergory: {:s}\tPixel-AUC: {:.6f}'
                  '\tImage-AUC: {:.6f}'.format(epoch, category, auc_pixel,
                                               auc_image_max))
        else:
            print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}'.
                  format(category, auc_pixel, auc_image_max))


if __name__ == "__main__":
    main()
