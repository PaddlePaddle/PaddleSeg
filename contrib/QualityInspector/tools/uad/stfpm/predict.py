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

import numpy as np
from PIL import Image

import paddle
import paddle.nn.functional as F
from paddle.vision import transforms

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
from qinspector.uad.models.stfpm import ResNet_MS3
from qinspector.cvlib.uad_configs import ConfigParser
from qinspector.uad.utils.utils import plot_fig


def argsparser():
    parser = argparse.ArgumentParser("STFPM")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--img_path", type=str, default=None, help="picture path")
    parser.add_argument('--resize', type=list, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default=None, help="student model_path")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_pic", type=bool, default=None)

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

    saved_dict = paddle.load(args.model_path)
    print('load ' + args.model_path)
    student.load_dict(saved_dict)

    img = Image.open(args.img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    teacher.eval()
    student.eval()
    with paddle.no_grad():
        t_feat = teacher(img)
        s_feat = student(img)
    score_map = 1.
    for j in range(len(t_feat)):
        t_feat[j] = F.normalize(t_feat[j], axis=1)
        s_feat[j] = F.normalize(s_feat[j], axis=1)
        sm = paddle.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
        sm = F.interpolate(
            sm,
            size=(args.resize[0], args.resize[1]),
            mode='bilinear',
            align_corners=False)
        # aggregate score map by element-wise product
        score_map = score_map * sm  # layer map

    if args.save_pic:
        save_name = os.path.join(args.save_path, args.category)
        dir_name = os.path.dirname(save_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plot_fig(img.numpy(),
                 score_map.squeeze(1), None, args.threshold, save_name,
                 args.category, args.save_pic, 'predict')

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Predict :  Picture {}".format(args.img_path) + " done!")


if __name__ == "__main__":
    main()
