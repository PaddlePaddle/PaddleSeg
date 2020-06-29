# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import math
import cv2
import paddle.fluid as fluid
import utils.logging as logging
from collections import OrderedDict
from .base import BaseModel
from utils.metrics import ConfusionMatrix
import nets


class UNet(BaseModel):
    """实现UNet网络的构建并进行训练、评估、预测和模型导出。

    Args:
        num_classes (int): 类别数。
        upsample_mode (str): UNet decode时采用的上采样方式，取值为'bilinear'时利用双线行差值进行上菜样，
            当输入其他选项时则利用反卷积进行上菜样，默认为'bilinear'。
        use_bce_loss (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。默认False。
        use_dice_loss (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。
            当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。默认False。
        class_weight (list/str): 交叉熵损失函数各类损失的权重。当class_weight为list的时候，长度应为
            num_classes。当class_weight为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重
            自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，
            即平时使用的交叉熵损失函数。
        ignore_index (int): label上忽略的值，label为ignore_index的像素不参与损失函数的计算。默认255。

    Raises:
        ValueError: use_bce_loss或use_dice_loss为真且num_calsses > 2。
        ValueError: class_weight为list, 但长度不等于num_class。
            class_weight为str, 但class_weight.low()不等于dynamic。
        TypeError: class_weight不为None时，其类型不是list或str。
    """

    def __init__(self,
                 num_classes=2,
                 upsample_mode='bilinear',
                 input_channel=3,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 sync_bn=True):
        super().__init__(
            num_classes=num_classes,
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss,
            class_weight=class_weight,
            ignore_index=ignore_index,
            sync_bn=sync_bn)
        self.init_params = locals()
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        if class_weight is not None:
            if isinstance(class_weight, list):
                if len(class_weight) != num_classes:
                    raise ValueError(
                        "Length of class_weight should be equal to number of classes"
                    )
            elif isinstance(class_weight, str):
                if class_weight.lower() != 'dynamic':
                    raise ValueError(
                        "if class_weight is string, must be dynamic!")
            else:
                raise TypeError(
                    'Expect class_weight is a list or string but receive {}'.
                    format(type(class_weight)))
        self.num_classes = num_classes
        self.upsample_mode = upsample_mode
        self.input_channel = input_channel
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.labels = None
        # 若模型是从inference model加载进来的，无法调用训练接口进行训练
        self.trainable = True

    def build_net(self, mode='train'):
        model = nets.UNet(
            self.num_classes,
            mode=mode,
            upsample_mode=self.upsample_mode,
            input_channel=self.input_channel,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        elif mode == 'eval':
            outputs['loss'] = model_out[0]
            outputs['pred'] = model_out[1]
            outputs['label'] = model_out[2]
            outputs['mask'] = model_out[3]
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs

    def train(self,
              num_epochs,
              train_reader,
              train_batch_size=2,
              eval_reader=None,
              eval_best_metric='miou',
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights=None,
              resume_weights=None,
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              regularization_coeff=5e-4,
              use_vdl=False):
        """训练。

        Args:
            num_epochs (int): 训练迭代轮数。
            train_reader (readers): 训练数据读取器。
            train_batch_size (int): 训练数据batch大小。同时作为验证数据batch大小。默认2。
            eval_reader (readers): 边训边评估的评估数据读取器。
            eval_best_metric (str): 边训边评估保存最好模型的指标。默认为'kappa'。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
            save_dir (str): 模型保存路径。默认'output'。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为None，则不使用预训练模型。
            optimizer (paddle.fluid.optimizer): 优化器。当改参数为None时，使用默认的优化器：使用
                fluid.optimizer.Momentum优化方法，polynomial的学习率衰减策略。
            learning_rate (float): 默认优化器的初始学习率。默认0.01。
            lr_decay_power (float): 默认优化器学习率多项式衰减系数。默认0.9。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认False。

        Raises:
            ValueError: 模型从inference model进行加载。
        """
        super().train(
            num_epochs=num_epochs,
            train_reader=train_reader,
            train_batch_size=train_batch_size,
            eval_reader=eval_reader,
            eval_best_metric=eval_best_metric,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            resume_weights=resume_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            regularization_coeff=regularization_coeff,
            use_vdl=use_vdl)

    def evaluate(self,
                 eval_reader,
                 batch_size=1,
                 verbose=True,
                 epoch_id=None,
                 return_details=False):
        """评估。

        Args:
            eval_reader (readers): 评估数据读取器。
            batch_size (int): 评估时的batch大小。默认1。
            verbose (bool): 是否打印日志。默认True。
            epoch_id (int): 当前评估模型所在的训练轮数。
            return_details (bool): 是否返回详细信息。默认False。

        Returns:
            dict: 当return_details为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
                'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
            tuple (metrics, eval_details)：当return_details为True时，增加返回dict (eval_details)，
                包含关键字：'confusion_matrix'，表示评估的混淆矩阵。
        """
        self.arrange_transform(transforms=eval_reader.transforms, mode='eval')
        total_steps = math.ceil(eval_reader.num_samples * 1.0 / batch_size)
        conf_mat = ConfusionMatrix(self.num_classes, streaming=True)
        data_generator = eval_reader.generator(
            batch_size=batch_size, drop_last=False)
        if not hasattr(self, 'parallel_test_prog'):
            self.parallel_test_prog = fluid.CompiledProgram(
                self.test_prog).with_data_parallel(
                    share_vars_from=self.parallel_train_prog)
        batch_size_each_gpu = self._get_single_card_bs(batch_size)

        for step, data in enumerate(data_generator()):
            images = np.array([d[0] for d in data])
            images = images.astype(np.float32)

            labels = np.array([d[1] for d in data])
            num_samples = images.shape[0]
            if num_samples < batch_size:
                num_pad_samples = batch_size - num_samples
                pad_images = np.tile(images[0:1], (num_pad_samples, 1, 1, 1))
                images = np.concatenate([images, pad_images])
            feed_data = {'image': images}
            outputs = self.exe.run(
                self.parallel_test_prog,
                feed=feed_data,
                fetch_list=list(self.test_outputs.values()),
                return_numpy=True)
            pred = outputs[0]
            if num_samples < batch_size:
                pred = pred[0:num_samples]

            mask = labels != self.ignore_index
            conf_mat.calculate(pred=pred, label=labels, ignore=mask)
            _, iou = conf_mat.mean_iou()

            if verbose:
                logging.info("[EVAL] Epoch={}, Step={}/{}, iou={}".format(
                    epoch_id, step + 1, total_steps, iou))

        category_iou, miou = conf_mat.mean_iou()
        category_acc, macc = conf_mat.accuracy()
        precision, recall = conf_mat.precision_recall()

        metrics = OrderedDict(
            zip([
                'miou', 'category_iou', 'macc', 'category_acc', 'kappa',
                'precision', 'recall'
            ], [
                miou, category_iou, macc, category_acc,
                conf_mat.kappa(), precision, recall
            ]))
        if return_details:
            eval_details = {
                'confusion_matrix': conf_mat.confusion_matrix.tolist()
            }
            return metrics, eval_details
        return metrics

    def predict(self, im_file, transforms=None):
        """预测。
        Args:
            img_file(str): 预测图像路径。
            transforms(transforms): 数据预处理操作。

        Returns:
            np.ndarray: 预测结果灰度图。
        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is not None:
            self.arrange_transform(transforms=transforms, mode='test')
            im, im_info = transforms(im_file)
        else:
            self.arrange_transform(transforms=self.test_transforms, mode='test')
            im, im_info = self.test_transforms(im_file)
        im = im.astype(np.float32)
        im = np.expand_dims(im, axis=0)
        result = self.exe.run(
            self.test_prog,
            feed={'image': im},
            fetch_list=list(self.test_outputs.values()))
        pred = result[0]
        pred = np.squeeze(pred).astype(np.uint8)
        keys = list(im_info.keys())
        for k in keys[::-1]:
            if k == 'shape_before_resize':
                h, w = im_info[k][0], im_info[k][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            elif k == 'shape_before_padding':
                h, w = im_info[k][0], im_info[k][1]
                pred = pred[0:h, 0:w]

        return {'label_map': pred}
