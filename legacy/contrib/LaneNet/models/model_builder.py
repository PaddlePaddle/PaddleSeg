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
import sys
sys.path.append("..")
import struct

import paddle.fluid as fluid
from paddle.fluid.proto.framework_pb2 import VarType

from pdseg import solver
from utils.config import cfg
from pdseg.loss import multi_softmax_with_loss
from loss import discriminative_loss
from models.modeling import lanenet


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
    if model_name == 'lanenet':
        logits = lanenet.lanenet(image, class_num)
    else:
        raise Exception(
            "unknow model name, only support unet, deeplabv3p, icnet, pspnet, hrnet"
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

    with fluid.program_guard(main_prog, start_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.data(name='label', shape=grt_shape, dtype='int32')
            if cfg.MODEL.MODEL_NAME == 'lanenet':
                label_instance = fluid.data(
                    name='label_instance', shape=grt_shape, dtype='int32')
            mask = fluid.data(name='mask', shape=grt_shape, dtype='int32')

            # use DataLoader.from_generator when doing traning and evaluation
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=[image, label, label_instance, mask],
                    capacity=cfg.DATALOADER.BUF_SIZE,
                    iterable=False,
                    use_double_buffer=True)

            loss_type = cfg.SOLVER.LOSS
            if not isinstance(loss_type, list):
                loss_type = list(loss_type)

            logits = seg_model(image, class_num)

            if ModelPhase.is_train(phase):
                loss_valid = False
                valid_loss = []
                if cfg.MODEL.MODEL_NAME == 'lanenet':
                    embeding_logit = logits[1]
                    logits = logits[0]
                    disc_loss, _, _, l_reg = discriminative_loss(
                        embeding_logit, label_instance, 4, image_shape[2:], 0.5,
                        3.0, 1.0, 1.0, 0.001)

                if "softmax_loss" in loss_type:
                    weight = None
                    if cfg.MODEL.MODEL_NAME == 'lanenet':
                        weight = get_dynamic_weight(label)
                    seg_loss = multi_softmax_with_loss(logits, label, mask,
                                                       class_num, weight)
                    loss_valid = True
                    valid_loss.append("softmax_loss")

                if not loss_valid:
                    raise Exception(
                        "SOLVER.LOSS: {} is set wrong. it should "
                        "include one of (softmax_loss, bce_loss, dice_loss) at least"
                        " example: ['softmax_loss']".format(cfg.SOLVER.LOSS))

                invalid_loss = [x for x in loss_type if x not in valid_loss]
                if len(invalid_loss) > 0:
                    print(
                        "Warning: the loss {} you set is invalid. it will not be included in loss computed."
                        .format(invalid_loss))

                avg_loss = disc_loss + 0.00001 * l_reg + seg_loss

            #get pred result in original size
            if isinstance(logits, tuple):
                logit = logits[0]
            else:
                logit = logits

            if logit.shape[2:] != label.shape[2:]:
                logit = fluid.layers.resize_bilinear(logit, label.shape[2:])

            # return image input and logit output for inference graph prune
            if ModelPhase.is_predict(phase):
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                return image, logit

            if class_num == 1:
                out = sigmoid_to_softmax(logit)
                out = fluid.layers.transpose(out, [0, 2, 3, 1])
            else:
                out = fluid.layers.transpose(logit, [0, 2, 3, 1])

            pred = fluid.layers.argmax(out, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            if ModelPhase.is_visual(phase):
                if cfg.MODEL.MODEL_NAME == 'lanenet':
                    return pred, logits[1]
                if class_num == 1:
                    logit = sigmoid_to_softmax(logit)
                else:
                    logit = softmax(logit)
                return pred, logit

            accuracy, fp, fn = compute_metric(pred, label)
            if ModelPhase.is_eval(phase):
                return data_loader, pred, label, mask, accuracy, fp, fn

            if ModelPhase.is_train(phase):
                optimizer = solver.Solver(main_prog, start_prog)
                decayed_lr = optimizer.optimise(avg_loss)
                return data_loader, avg_loss, decayed_lr, pred, label, mask, disc_loss, seg_loss, accuracy, fp, fn


def compute_metric(pred, label):
    label = fluid.layers.transpose(label, [0, 2, 3, 1])

    idx = fluid.layers.where(pred == 1)
    pix_cls_ret = fluid.layers.gather_nd(label, idx)

    correct_num = fluid.layers.reduce_sum(
        fluid.layers.cast(pix_cls_ret, 'float32'))

    gt_num = fluid.layers.cast(
        fluid.layers.shape(
            fluid.layers.gather_nd(label, fluid.layers.where(label == 1)))[0],
        'int64')
    pred_num = fluid.layers.cast(
        fluid.layers.shape(fluid.layers.gather_nd(pred, idx))[0], 'int64')
    accuracy = correct_num / gt_num

    false_pred = pred_num - correct_num
    fp = fluid.layers.cast(false_pred, 'float32') / fluid.layers.cast(
        fluid.layers.shape(pix_cls_ret)[0], 'int64')

    label_cls_ret = fluid.layers.gather_nd(label,
                                           fluid.layers.where(label == 1))
    mis_pred = fluid.layers.cast(fluid.layers.shape(label_cls_ret)[0],
                                 'int64') - correct_num
    fn = fluid.layers.cast(mis_pred, 'float32') / fluid.layers.cast(
        fluid.layers.shape(label_cls_ret)[0], 'int64')
    accuracy.stop_gradient = True
    fp.stop_gradient = True
    fn.stop_gradient = True
    return accuracy, fp, fn


def get_dynamic_weight(label):
    label = fluid.layers.reshape(label, [-1])
    unique_labels, unique_id, counts = fluid.layers.unique_with_counts(label)
    counts = fluid.layers.cast(counts, 'float32')
    weight = 1.0 / fluid.layers.log(
        (counts / fluid.layers.reduce_sum(counts) + 1.02))
    return weight


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
