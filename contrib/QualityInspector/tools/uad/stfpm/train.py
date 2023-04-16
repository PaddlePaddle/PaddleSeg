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
import datetime
import argparse
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.vision import transforms

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
from qinspector.uad.models.stfpm import ResNet_MS3
from qinspector.uad.datasets.mvtec import MVTecDatasetSTFPM
from qinspector.cvlib.uad_configs import ConfigParser
from val import val, cal_error


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
    parser.add_argument("--do_eval", type=bool, default=None)
    parser.add_argument(
        "--epochs", type=int, default=None, help='number of epochs')
    parser.add_argument("--lr", type=float, default=None, help='learning rate')
    parser.add_argument("--momentum", type=float, default=None, help='momentum')
    parser.add_argument(
        "--weight_decay", type=float, default=None, help='weight_decay')

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
    teacher = ResNet_MS3(arch=args.backbone, pretrained=True)
    student = ResNet_MS3(arch=args.backbone, pretrained=False)

    # build datasets
    transform = transforms.Compose([
        transforms.Resize(args.resize), transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_list = sorted(
        glob(
            os.path.join(args.data_path, args.category, 'train', 'good',
                         '*.png')))
    train_image_list, val_image_list = train_test_split(
        image_list, test_size=0.2, random_state=0)
    train_dataset = MVTecDatasetSTFPM(train_image_list, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    val_dataset = MVTecDatasetSTFPM(val_image_list, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)
    if args.do_eval:
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

    if args.do_eval:
        train(teacher, student, train_loader, val_loader, args, test_pos_loader,
              test_neg_loader)
    else:
        train(teacher, student, train_loader, val_loader, args)


def train(teacher,
          student,
          train_loader,
          val_loader,
          args,
          test_pos_loader=None,
          test_neg_loader=None):
    min_err = 10000
    teacher.eval()
    student.train()
    optimizer = optim.Momentum(
        parameters=student.parameters(),
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        student.train()
        epoch_begin = time.time()
        end_time = time.time()
        for index, batch_data in enumerate(train_loader):
            optimizer.clear_grad()
            batch_begin = time.time()
            data_time = batch_begin - end_time
            _, batch_img = batch_data

            with paddle.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss = paddle.to_tensor(0.0)
            loss.stop_gradient = False
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], axis=1)
                s_feat[i] = F.normalize(s_feat[i], axis=1)
                loss += paddle.sum((t_feat[i] - s_feat[i])**2, 1).mean()

            loss.backward()
            optimizer.step()
            lr = optimizer.get_lr()
            end_time = time.time()
            bacth_time = end_time - batch_begin

            if index % args.print_freq == 0:
                print(
                    datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S") + '\t' +
                    "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".
                    format(epoch, args.batch_size * (index + 1),
                           len(train_loader.dataset),
                           loss.cpu().numpy()[0],
                           float(lr), float(bacth_time), float(data_time)))
        t = time.time() - epoch_begin
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Epoch {} training ends, total {:.2f}s".format(epoch, t))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Epoch {} testing start".format(epoch))
        err = cal_error(args, teacher, student, val_loader).mean()
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              'Valid Loss: {:.7f}'.format(err.item()))
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.save_path, args.category,
                                     'best.pdparams')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            paddle.save(student.state_dict(), save_name)
        if args.do_eval:
            val(args,
                student,
                teacher,
                test_pos_loader,
                test_neg_loader,
                epoch=epoch,
                eval_pro=args.compute_pro)
        t = time.time() - epoch_begin - t
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
              "Epoch {} testing end, total {:.2f}s".format(epoch, t))


if __name__ == "__main__":
    main()
