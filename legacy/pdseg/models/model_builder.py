# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import struct

import paddle
import paddle.static as static
import paddle.nn.functional as F
import numpy as np
from paddle.fluid.proto.framework_pb2 import VarType

import solver
from utils.config import cfg
from loss import multi_softmax_with_loss
from loss import multi_dice_loss
from loss import multi_bce_loss
from lovasz_losses import multi_lovasz_hinge_loss, multi_lovasz_softmax_loss
from models.modeling import deeplab, unet, icnet, pspnet, hrnet, fast_scnn, ocrnet


class ModelPhase(object):
    """
    Standard name for model phase in PaddleSeg

    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `EVAL`: testing/evaluation mode.
    * `PREDICT`: prediction/inference mode.
    * `VISUAL` : visualization mode
    """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'
    VISUAL = 'visual'

    @staticmethod
    def is_train(phase):
        return phase == ModelPhase.TRAIN

    @staticmethod
    def is_predict(phase):
        return phase == ModelPhase.PREDICT

    @staticmethod
    def is_eval(phase):
        return phase == ModelPhase.EVAL

    @staticmethod
    def is_visual(phase):
        return phase == ModelPhase.VISUAL

    @staticmethod
    def is_valid_phase(phase):
        """ Check valid phase """
        if ModelPhase.is_train(phase) or ModelPhase.is_predict(phase) \
                or ModelPhase.is_eval(phase) or ModelPhase.is_visual(phase):
            return True

        return False


def seg_model(image, class_num):
    model_name = cfg.MODEL.MODEL_NAME
    if model_name == 'unet':
        logits = unet.unet(image, class_num)
    elif model_name == 'deeplabv3p':
        logits = deeplab.deeplabv3p(image, class_num)
    elif model_name == 'icnet':
        logits = icnet.icnet(image, class_num)
    elif model_name == 'pspnet':
        logits = pspnet.pspnet(image, class_num)
    elif model_name == 'hrnet':
        logits = hrnet.hrnet(image, class_num)
    elif model_name == 'fast_scnn':
        logits = fast_scnn.fast_scnn(image, class_num)
    elif model_name == 'ocrnet':
        logits = ocrnet.ocrnet(image, class_num)
    else:
        raise Exception(
            "unknow model name, only support unet, deeplabv3p, icnet, pspnet, hrnet, fast_scnn"
        )
    return logits


def softmax(logit):
    logit = paddle.transpose(logit, [0, 2, 3, 1])
    logit = F.softmax(logit)
    logit = paddle.transpose(logit, [0, 3, 1, 2])
    return logit


def sigmoid_to_softmax(logit):
    """
    one channel to two channel
    """
    logit = paddle.transpose(logit, [0, 2, 3, 1])
    logit = F.sigmoid(logit)
    logit_back = 1 - logit
    logit = paddle.concat([logit_back, logit], axis=-1)
    logit = paddle.transpose(logit, [0, 3, 1, 2])
    return logit


def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    if not ModelPhase.is_valid_phase(phase):
        raise ValueError("ModelPhase {} is not valid!".format(phase))
    if ModelPhase.is_train(phase):
        width = cfg.TRAIN_CROP_SIZE[0]
        height = cfg.TRAIN_CROP_SIZE[1]
    else:
        width = cfg.EVAL_CROP_SIZE[0]
        height = cfg.EVAL_CROP_SIZE[1]

    image_shape = [-1, cfg.DATASET.DATA_DIM, height, width]
    grt_shape = [-1, 1, height, width]
    class_num = cfg.DATASET.NUM_CLASSES

    with static.program_guard(main_prog, start_prog):
        _new_generator = paddle.fluid.unique_name.UniqueNameGenerator()
        with paddle.utils.unique_name.guard(_new_generator):
            # 在导出模型的时候，增加图像标准化预处理,减小预测部署时图像的处理流程
            # 预测部署时只须对输入图像增加batch_size维度即可
            image = static.data(
                name='image', shape=image_shape, dtype='float32')
            label = static.data(name='label', shape=grt_shape, dtype='int32')
            mask = static.data(name='mask', shape=grt_shape, dtype='int32')

            # use DataLoader when doing traning and evaluation
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                data_loader = paddle.io.DataLoader.from_generator(
                    feed_list=[image, label, mask],
                    capacity=cfg.DATALOADER.BUF_SIZE,
                    iterable=False,
                    use_double_buffer=True)

            loss_type = cfg.SOLVER.LOSS
            if not isinstance(loss_type, list):
                loss_type = list(loss_type)

            # lovasz_hinge_loss或dice_loss或bce_loss只适用两类分割中
            if class_num > 2 and (("lovasz_hinge_loss" in loss_type) or
                                  ("dice_loss" in loss_type) or
                                  ("bce_loss" in loss_type)):
                raise Exception(
                    "lovasz hinge loss, dice loss and bce loss are only applicable to binary classfication."
                )

            # 在两类分割情况下，当loss函数选择lovasz_hinge_loss或dice_loss或bce_loss的时候，最后logit输出通道数设置为1
            if ("dice_loss" in loss_type) or ("bce_loss" in loss_type) or (
                    "lovasz_hinge_loss" in loss_type):
                class_num = 1
                if ("softmax_loss" in loss_type) or (
                        "lovasz_softmax_loss" in loss_type):
                    raise Exception(
                        "softmax loss or lovasz softmax loss can not combine with bce loss or dice loss or lovasz hinge loss."
                    )
            cfg.PHASE = phase
            logits = seg_model(image, class_num)

            # 根据选择的loss函数计算相应的损失函数
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                loss_valid = False
                avg_loss_list = []
                valid_loss = []
                if "softmax_loss" in loss_type:
                    weight = cfg.SOLVER.CROSS_ENTROPY_WEIGHT
                    avg_loss_list.append(
                        multi_softmax_with_loss(logits, label, mask, class_num,
                                                weight))
                    loss_valid = True
                    valid_loss.append("softmax_loss")
                if "dice_loss" in loss_type:
                    avg_loss_list.append(multi_dice_loss(logits, label, mask))
                    loss_valid = True
                    valid_loss.append("dice_loss")
                if "bce_loss" in loss_type:
                    avg_loss_list.append(multi_bce_loss(logits, label, mask))
                    loss_valid = True
                    valid_loss.append("bce_loss")
                if "lovasz_hinge_loss" in loss_type:
                    avg_loss_list.append(
                        multi_lovasz_hinge_loss(logits, label, mask))
                    loss_valid = True
                    valid_loss.append("lovasz_hinge_loss")
                if "lovasz_softmax_loss" in loss_type:
                    avg_loss_list.append(
                        multi_lovasz_softmax_loss(logits, label, mask))
                    loss_valid = True
                    valid_loss.append("lovasz_softmax_loss")
                if not loss_valid:
                    raise Exception(
                        "SOLVER.LOSS: {} is set wrong. it should "
                        "include one of (softmax_loss, bce_loss, dice_loss, lovasz_hinge_loss, lovasz_softmax_loss) at least"
                        " example: ['softmax_loss'], ['dice_loss'], ['bce_loss', 'dice_loss'], ['lovasz_hinge_loss','bce_loss'], ['lovasz_softmax_loss','softmax_loss']"
                        .format(cfg.SOLVER.LOSS))

                invalid_loss = [x for x in loss_type if x not in valid_loss]
                if len(invalid_loss) > 0:
                    print(
                        "Warning: the loss {} you set is invalid. it will not be included in loss computed."
                        .format(invalid_loss))

                avg_loss = 0
                for i in range(0, len(avg_loss_list)):
                    loss_name = valid_loss[i].upper()
                    loss_weight = eval('cfg.SOLVER.LOSS_WEIGHT.' + loss_name)
                    avg_loss += loss_weight * avg_loss_list[i]

            #get pred result in original size
            if isinstance(logits, tuple):
                logit = logits[0]
            else:
                logit = logits

            if logit.shape[2:] != label.shape[2:]:
                logit = F.interpolate(
                    logit,
                    label.shape[2:],
                    mode='bilinear',
                    align_corners=False)

            # return image input and logit output for inference graph prune
            if ModelPhase.is_predict(phase):
                # 两类分割中，使用lovasz_hinge_loss或dice_loss或bce_loss返回的logit为单通道，进行到两通道的变换
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)

                return image, logit

            if class_num == 1:
                out = sigmoid_to_softmax(logit)
                out = paddle.transpose(out, [0, 2, 3, 1])
            else:
                out = paddle.transpose(logit, [0, 2, 3, 1])

            pred = paddle.argmax(out, axis=3)
            pred = paddle.unsqueeze(pred, axis=[3])
            if ModelPhase.is_visual(phase):
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                return pred, logit

            if ModelPhase.is_eval(phase):
                return data_loader, avg_loss, pred, label, mask

            if ModelPhase.is_train(phase):
                optimizer = solver.Solver(main_prog, start_prog)
                decayed_lr, optimizer_ = optimizer.optimise(avg_loss)
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                return data_loader, avg_loss, decayed_lr, pred, label, mask, optimizer_, _new_generator


def to_int(string, dest="I"):
    return struct.unpack(dest, string)[0]


def parse_shape_from_file(filename):
    with open(filename, "rb") as file:
        version = file.read(4)
        lod_level = to_int(file.read(8), dest="Q")
        for i in range(lod_level):
            _size = to_int(file.read(8), dest="Q")
            _ = file.read(_size)
        version = file.read(4)
        tensor_desc_size = to_int(file.read(4))
        tensor_desc = VarType.TensorDesc()
        tensor_desc.ParseFromString(file.read(tensor_desc_size))
    return tuple(tensor_desc.dims)
