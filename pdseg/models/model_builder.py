# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import struct
import importlib

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.proto.framework_pb2 import VarType

import solver
from utils.config import cfg
from loss import multi_softmax_with_loss
from loss import multi_dice_loss
from loss import multi_bce_loss


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


def map_model_name(model_name):
    name_dict = {
        "unet": "unet.unet",
        "deeplabv3p": "deeplab.deeplabv3p",
        "icnet": "icnet.icnet",
        "pspnet": "pspnet.pspnet",
        "hrnet": "hrnet.hrnet"
    }
    if model_name in name_dict.keys():
        return name_dict[model_name]
    else:
        raise Exception(
            "unknow model name, only support unet, deeplabv3p, icnet")


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'models.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to find function: {}'.format(func_name))
    return module


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


def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    if not ModelPhase.is_valid_phase(phase):
        raise ValueError("ModelPhase {} is not valid!".format(phase))
    if ModelPhase.is_train(phase):
        width = cfg.TRAIN_CROP_SIZE[0]
        height = cfg.TRAIN_CROP_SIZE[1]
    else:
        width = cfg.EVAL_CROP_SIZE[0]
        height = cfg.EVAL_CROP_SIZE[1]

    image_shape = [cfg.DATASET.DATA_DIM, height, width]
    grt_shape = [1, height, width]
    class_num = cfg.DATASET.NUM_CLASSES

    with fluid.program_guard(main_prog, start_prog):
        with fluid.unique_name.guard():
            # 在导出模型的时候，增加图像标准化预处理,减小预测部署时图像的处理流程
            # 预测部署时只须对输入图像增加batch_size维度即可
            if ModelPhase.is_predict(phase):
                origin_image = fluid.layers.data(
                    name='image',
                    shape=[-1, 1, 1, cfg.DATASET.DATA_DIM],
                    dtype='float32',
                    append_batch_size=False)
                image = fluid.layers.transpose(origin_image, [0, 3, 1, 2])
                origin_shape = fluid.layers.shape(image)[-2:]
                mean = np.array(cfg.MEAN).reshape(1, len(cfg.MEAN), 1, 1)
                mean = fluid.layers.assign(mean.astype('float32'))
                std = np.array(cfg.STD).reshape(1, len(cfg.STD), 1, 1)
                std = fluid.layers.assign(std.astype('float32'))
                image = fluid.layers.resize_bilinear(
                    image,
                    out_shape=[height, width],
                    align_corners=False,
                    align_mode=0)
                image = (image / 255 - mean) / std
            else:
                image = fluid.layers.data(
                    name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(
                name='label', shape=grt_shape, dtype='int32')
            mask = fluid.layers.data(
                name='mask', shape=grt_shape, dtype='int32')

            # use PyReader when doing traning and evaluation
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                py_reader = fluid.io.PyReader(
                    feed_list=[image, label, mask],
                    capacity=cfg.DATALOADER.BUF_SIZE,
                    iterable=False,
                    use_double_buffer=True)

            model_name = map_model_name(cfg.MODEL.MODEL_NAME)
            model_func = get_func("modeling." + model_name)

            loss_type = cfg.SOLVER.LOSS
            if not isinstance(loss_type, list):
                loss_type = list(loss_type)

            # dice_loss或bce_loss只适用两类分割中
            if class_num > 2 and (("dice_loss" in loss_type) or
                                  ("bce_loss" in loss_type)):
                raise Exception(
                    "dice loss and bce loss is only applicable to binary classfication"
                )

            # 在两类分割情况下，当loss函数选择dice_loss或bce_loss的时候，最后logit输出通道数设置为1
            if ("dice_loss" in loss_type) or ("bce_loss" in loss_type):
                class_num = 1
                if "softmax_loss" in loss_type:
                    raise Exception(
                        "softmax loss can not combine with dice loss or bce loss"
                    )

            logits = model_func(image, class_num)

            # 根据选择的loss函数计算相应的损失函数
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                loss_valid = False
                avg_loss_list = []
                valid_loss = []
                if "softmax_loss" in loss_type:
                    avg_loss_list.append(
                        multi_softmax_with_loss(logits, label, mask, class_num))
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
                if not loss_valid:
                    raise Exception(
                        "SOLVER.LOSS: {} is set wrong. it should "
                        "include one of (softmax_loss, bce_loss, dice_loss) at least"
                        " example: ['softmax_loss'], ['dice_loss'], ['bce_loss', 'dice_loss']"
                        .format(cfg.SOLVER.LOSS))

                invalid_loss = [x for x in loss_type if x not in valid_loss]
                if len(invalid_loss) > 0:
                    print(
                        "Warning: the loss {} you set is invalid. it will not be included in loss computed."
                        .format(invalid_loss))

                avg_loss = 0
                for i in range(0, len(avg_loss_list)):
                    avg_loss += avg_loss_list[i]

            #get pred result in original size
            if isinstance(logits, tuple):
                logit = logits[0]
            else:
                logit = logits

            if logit.shape[2:] != label.shape[2:]:
                logit = fluid.layers.resize_bilinear(logit, label.shape[2:])

            # return image input and logit output for inference graph prune
            if ModelPhase.is_predict(phase):
                # 两类分割中，使用dice_loss或bce_loss返回的logit为单通道，进行到两通道的变换
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                logit = fluid.layers.resize_bilinear(
                    logit,
                    out_shape=origin_shape,
                    align_corners=False,
                    align_mode=0)
                logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
                logit = fluid.layers.argmax(logit, axis=3)
                return origin_image, logit

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
                return pred, logit

            if ModelPhase.is_eval(phase):
                return py_reader, avg_loss, pred, label, mask

            if ModelPhase.is_train(phase):
                optimizer = solver.Solver(main_prog, start_prog)
                decayed_lr = optimizer.optimise(avg_loss)
                return py_reader, avg_loss, decayed_lr, pred, label, mask


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
