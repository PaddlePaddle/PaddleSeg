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

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.proto.framework_pb2 import VarType

import solver
from utils.config import cfg
from loss import multi_softmax_with_loss
from loss import multi_dice_loss
from loss import multi_bce_loss
from lovasz_losses import lovasz_hinge
from lovasz_losses import lovasz_softmax
from models.modeling import deeplab, unet, icnet, pspnet, hrnet, fast_scnn
from models.libs.model_libs import fuse


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
    else:
        raise Exception(
            "unknow model name, only support unet, deeplabv3p, icnet, pspnet, hrnet, fast_scnn"
        )
    return logits


def softmax(logit):
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.softmax(logit)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit


def sigmoid_to_softmax(logit):
    """
    one channel to two channel
    """
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.sigmoid(logit)
    logit_back = 1 - logit
    logit = fluid.layers.concat([logit_back, logit], axis=-1)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit


def test_aug_inv(logits_split):
    ops = cfg.TEST.TEST_AUG_FLIP_OPS + cfg.TEST.TEST_AUG_ROTATE_OPS
    for i, logits in enumerate(logits_split[1:]):
        if type(ops[i]) is str:
            if ops[i][0] == 'h':
                logits_split[i + 1] = fluid.layers.flip(logits, [2])
            elif ops[i][0] == 'v':
                logits_split[i + 1] = fluid.layers.flip(logits, [3])
            elif ops[i][0] == 'm':
                logits_split[i + 1] = fluid.layers.transpose(
                    logits, [0, 1, 3, 2])
            else:
                rot90 = fluid.layers.flip(
                    fluid.layers.transpose(logits, [0, 1, 3, 2]), [2])
                rot180 = fluid.layers.flip(
                    fluid.layers.transpose(rot90, [0, 1, 3, 2]), [2])
                logits_split[i + 1] = fluid.layers.transpose(
                    rot180, [0, 1, 3, 2])
        else:
            times = (360 - ops[i]) // 90
            for _ in range(times):
                logits = fluid.layers.flip(
                    fluid.layers.transpose(logits, [0, 1, 3, 2]), [2])
            logits_split[i + 1] = fluid.layers.transpose(logits, [0, 1, 3, 2])

    logits = fluid.layers.stack(
        logits_split, axis=0)  # channel_mul, batch_size, 1, h, w
    logits = fluid.layers.reduce_mean(logits, dim=0)  # batch_size, 1, h, w
    return logits


def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    if not ModelPhase.is_valid_phase(phase):
        raise ValueError("ModelPhase {} is not valid!".format(phase))
    if ModelPhase.is_train(phase):
        width = cfg.TRAIN_CROP_SIZE[0]
        height = cfg.TRAIN_CROP_SIZE[1]
    else:
        width = cfg.EVAL_CROP_SIZE[0]
        height = cfg.EVAL_CROP_SIZE[1]

    channel_mul = 1
    if ModelPhase.is_eval(phase) and cfg.TEST.TEST_AUG:
        channel_mul = len(cfg.TEST.TEST_AUG_FLIP_OPS) + len(
            cfg.TEST.TEST_AUG_ROTATE_OPS) + 1
    image1_shape = [-1, cfg.DATASET.DATA_DIM * channel_mul, height, width]

    grt_shape = [-1, 1, height, width]
    class_num = cfg.DATASET.NUM_CLASSES

    with fluid.program_guard(main_prog, start_prog):
        with fluid.unique_name.guard():
            # 在导出模型的时候，增加图像标准化预处理,减小预测部署时图像的处理流程
            # 预测部署时只须对输入图像增加batch_size维度即可
            image1 = fluid.data(
                name='image1', shape=image1_shape, dtype='float32')
            image2 = None
            if cfg.DATASET.INPUT_IMAGE_NUM == 2:
                image2_shape = [
                    -1, cfg.DATASET.DATA_DIM * channel_mul, height, width
                ]
                image2 = fluid.data(
                    name='image2', shape=image2_shape, dtype='float32')
            label = fluid.data(name='label', shape=grt_shape, dtype='int32')
            mask = fluid.data(name='mask', shape=grt_shape, dtype='int32')

            # use DataLoader when doing traning and evaluation
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                if cfg.DATASET.INPUT_IMAGE_NUM == 1:
                    data_loader = fluid.io.DataLoader.from_generator(
                        feed_list=[image1, label, mask],
                        capacity=cfg.DATALOADER.BUF_SIZE,
                        iterable=False,
                        use_double_buffer=True)
                else:
                    data_loader = fluid.io.DataLoader.from_generator(
                        feed_list=[image1, image2, label, mask],
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

            if cfg.DATASET.INPUT_IMAGE_NUM == 1:
                if cfg.TEST.TEST_AUG:
                    image_split = fluid.layers.split(
                        image1, channel_mul,
                        dim=1)  # batch_size, 3 x channel_mul, h, w
                    image1 = fluid.layers.concat(
                        image_split,
                        axis=0)  # channel_mul * batch_size, 3, h, w
                    logits = seg_model(image1, class_num)
                    logits_split = fluid.layers.split(
                        logits, channel_mul, dim=0)
                    logits = test_aug_inv(logits_split)
                else:
                    logits = seg_model(image1, class_num)

            else:
                if cfg.TEST.TEST_AUG:
                    image_split = fluid.layers.split(
                        image1, channel_mul,
                        dim=1)  # batch_size, 3 x channel_mul, h, w
                    image1 = fluid.layers.concat(
                        image_split,
                        axis=0)  # channel_mul * batch_size, 3, h, w
                    logits1 = seg_model(image1, class_num)
                    logits1_split = fluid.layers.split(
                        logits1, channel_mul, dim=0)
                    logits1 = test_aug_inv(logits1_split)

                    image_split = fluid.layers.split(
                        image2, channel_mul,
                        dim=1)  # batch_size, 3 x channel_mul, h, w
                    image2 = fluid.layers.concat(
                        image_split,
                        axis=0)  # channel_mul * batch_size, 3, h, w
                    logits2 = seg_model(image2, class_num)
                    logits2_split = fluid.layers.split(
                        logits2, channel_mul, dim=0)
                    logits2 = test_aug_inv(logits2_split)
                else:
                    logits1 = seg_model(image1, class_num)
                    logits2 = seg_model(image2, class_num)

                if ModelPhase.is_visual(phase) and cfg.VIS.SEG_FOR_CD:
                    logits = fluid.layers.concat([logits1, logits2], axis=0)
                else:
                    logits = fluid.layers.concat([logits1, logits2], axis=1)
                    logits = fuse(logits, class_num)

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
                        lovasz_hinge(logits, label, ignore=mask))
                    loss_valid = True
                    valid_loss.append("lovasz_hinge_loss")
                if "lovasz_softmax_loss" in loss_type:
                    probas = fluid.layers.softmax(logits, axis=1)
                    avg_loss_list.append(
                        lovasz_softmax(probas, label, ignore=mask))
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
                logit = fluid.layers.resize_bilinear(logit, label.shape[2:])

            # return image input and logit output for inference graph prune
            if ModelPhase.is_predict(phase):
                # 两类分割中，使用lovasz_hinge_loss或dice_loss或bce_loss返回的logit为单通道，进行到两通道的变换
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)

                if cfg.DATASET.INPUT_IMAGE_NUM == 1:
                    return image1, logit
                else:
                    return image1, image2, logit

            if class_num == 1:
                out = sigmoid_to_softmax(logit)
                out = fluid.layers.transpose(out, [0, 2, 3, 1])
            else:
                out = fluid.layers.transpose(logit, [0, 2, 3, 1])

            pred = fluid.layers.argmax(out, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])

            if ModelPhase.is_visual(phase):
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                    pred, _ = fluid.layers.split(logit, 2, dim=1)
                return pred, logit

            if ModelPhase.is_eval(phase):
                if cfg.VIS.VISINEVAL:
                    logit = softmax(logit)
                    pred, _ = fluid.layers.split(logit, 2, dim=1)
                return data_loader, avg_loss, pred, label, mask

            if ModelPhase.is_train(phase):
                optimizer = solver.Solver(main_prog, start_prog)
                decayed_lr = optimizer.optimise(avg_loss)
                return data_loader, avg_loss, decayed_lr, pred, label, mask


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
