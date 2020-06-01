# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
import paddle.fluid as fluid
import os
from os import path as osp
import numpy as np
from collections import OrderedDict
import copy
import math
import time
import tqdm
import cv2
import yaml
import shutil

from paddle.fluid.dygraph.base import to_variable

import utils
import utils.logging as logging
from utils import seconds_to_hms
from utils import ConfusionMatrix
from utils import get_environ_info
import nets
import transforms as T


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


class UNet(object):
    # DeepLab mobilenet
    def __init__(self,
                 num_classes=2,
                 upsample_mode='bilinear',
                 ignore_index=255):

        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        self.ignore_index = ignore_index

        self.labels = None
        self.env_info = get_environ_info()
        if self.env_info['place'] == 'cpu':
            self.places = fluid.CPUPlace()
        else:
            self.places = fluid.CUDAPlace(0)

    def build_model(self):
        self.model = nets.UNet(self.num_classes, self.upsample_mode)

    def arrange_transform(self, transforms, mode='train'):
        arrange_transform = T.ArrangeSegmenter
        if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
            transforms.transforms[-1] = arrange_transform(mode=mode)
        else:
            transforms.transforms.append(arrange_transform(mode=mode))

    def load_model(self, model_dir):
        ckpt_path = osp.join(model_dir, 'model')
        para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
        self.model.set_dict(para_state_dict)

    def save_model(self, state_dict, save_dir):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        fluid.save_dygraph(state_dict, osp.join(save_dir, 'model'))

    def default_optimizer(self,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          parameter_list=None,
                          lr_decay_power=0.9,
                          regularization_coeff=4e-5):
        decay_step = num_epochs * num_steps_each_epoch
        lr_decay = fluid.layers.polynomial_decay(
            learning_rate,
            decay_step,
            end_learning_rate=0,
            power=lr_decay_power)
        optimizer = fluid.optimizer.Momentum(
            lr_decay,
            momentum=0.9,
            parameter_list=parameter_list,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=regularization_coeff))
        return optimizer

    def _get_loss(self, logit, label):
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + 0.00001)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrained_weights=None,
              resume_weights=None,
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              regularization_coeff=4e-5,
              use_vdl=False):
        self.labels = train_dataset.labels
        self.train_transforms = train_dataset.transforms
        self.train_init = locals()
        self.begin_epoch = 0
        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            optimizer = self.default_optimizer(
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                num_steps_each_epoch=num_steps_each_epoch,
                parameter_list=self.model.parameters(),
                lr_decay_power=lr_decay_power,
                regularization_coeff=regularization_coeff)

        # to do: 预训练模型加载， resume

        if self.begin_epoch >= num_epochs:
            raise ValueError(
                ("begin epoch[{}] is larger than num_epochs[{}]").format(
                    self.begin_epoch, num_epochs))

        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)

        # add arrange op to transforms
        self.arrange_transform(
            transforms=train_dataset.transforms, mode='train')

        if eval_dataset is not None:
            self.eval_transforms = eval_dataset.transforms
            self.test_transforms = copy.deepcopy(eval_dataset.transforms)

        data_generator = train_dataset.generator(
            batch_size=train_batch_size, drop_last=True)
        total_num_steps = math.floor(
            train_dataset.num_samples / train_batch_size)

        for i in range(self.begin_epoch, num_epochs):
            for step, data in enumerate(data_generator()):
                images = np.array([d[0] for d in data])
                labels = np.array([d[1] for d in data]).astype('int64')
                images = to_variable(images)
                labels = to_variable(labels)
                logit = self.model(images)
                loss = self._get_loss(logit, labels)
                loss.backward()
                optimizer.minimize(loss)
                print("[TRAIN] Epoch={}/{}, Step={}/{}, loss={}".format(
                    i + 1, num_epochs, step + 1, total_num_steps, loss.numpy()))

            if (i + 1) % save_interval_epochs == 0 or i == num_epochs - 1:
                current_save_dir = osp.join(save_dir, "epoch_{}".format(i + 1))
                if not osp.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                self.save_model(self.model.state_dict(), current_save_dir)
                if eval_dataset is not None:
                    self.model.eval()
                    self.evaluate(eval_dataset, batch_size=train_batch_size)
                    self.model.train()

    def evaluate(self, eval_dataset, batch_size=1, epoch_id=None):
        """评估。

        Args:
            eval_dataset (paddlex.datasets): 评估数据读取器。
            batch_size (int): 评估时的batch大小。默认1。
            epoch_id (int): 当前评估模型所在的训练轮数。
            return_details (bool): 是否返回详细信息。默认False。

        Returns:
            dict: 当return_details为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
                'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
            tuple (metrics, eval_details)：当return_details为True时，增加返回dict (eval_details)，
                包含关键字：'confusion_matrix'，表示评估的混淆矩阵。
        """
        self.model.eval()
        self.arrange_transform(transforms=eval_dataset.transforms, mode='train')
        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        conf_mat = ConfusionMatrix(self.num_classes, streaming=True)
        data_generator = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)
        logging.info(
            "Start to evaluating(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(
                enumerate(data_generator()), total=total_steps):
            images = np.array([d[0] for d in data])
            labels = np.array([d[1] for d in data])
            images = to_variable(images)

            logit = self.model(images)
            pred = fluid.layers.argmax(logit, axis=1)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            pred = pred.numpy()

            mask = labels != self.ignore_index
            conf_mat.calculate(pred=pred, label=labels, ignore=mask)
            _, iou = conf_mat.mean_iou()

            logging.debug("[EVAL] Epoch={}, Step={}/{}, iou={}".format(
                epoch_id, step + 1, total_steps, iou))

        category_iou, miou = conf_mat.mean_iou()
        category_acc, macc = conf_mat.accuracy()

        metrics = OrderedDict(
            zip(['miou', 'category_iou', 'macc', 'category_acc', 'kappa'],
                [miou, category_iou, macc, category_acc,
                 conf_mat.kappa()]))

        logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
            epoch_id, dict2str(metrics)))
        return metrics

    def predict(self, im_file, transforms=None):
        """预测。
        Args:
            img_file(str|np.ndarray): 预测图像。
            transforms(paddlex.cv.transforms): 数据预处理操作。

        Returns:
            dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """
        if isinstance(im_file, str):
            if not osp.exists(im_file):
                raise ValueError(
                    'The Image file does not exist: {}'.format(im_file))

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is not None:
            self.arrange_transform(transforms=transforms, mode='test')
            im, im_info = transforms(im_file)
        else:
            self.arrange_transform(transforms=self.test_transforms, mode='test')
            im, im_info = self.test_transforms(im_file)
        im = np.expand_dims(im, axis=0)
        im = to_variable(im)
        logit = self.model(im)
        logit = fluid.layers.softmax(logit)
        pred = fluid.layers.argmax(logit, axis=1)
        logit = logit.numpy()
        pred = pred.numpy()

        logit = np.squeeze(logit)
        logit = np.transpose(logit, (1, 2, 0))
        pred = np.squeeze(pred).astype('uint8')
        keys = list(im_info.keys())
        print(pred.shape, logit.shape)
        for k in keys[::-1]:
            if k == 'shape_before_resize':
                h, w = im_info[k][0], im_info[k][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
                logit = cv2.resize(logit, (w, h), cv2.INTER_LINEAR)
            elif k == 'shape_before_padding':
                h, w = im_info[k][0], im_info[k][1]
                pred = pred[0:h, 0:w]
                logit = logit[0:h, 0:w, :]

        return {'label_map': pred, 'score_map': logit}
