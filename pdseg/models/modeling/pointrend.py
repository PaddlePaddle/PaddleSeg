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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, avg_pool, conv, bn_relu, relu
from models.libs.model_libs import separate_conv


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


def mlp(input, num_classes):
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    with scope('mlp'):
        with scope('conv1'):
            data = relu(
                conv(
                    input,
                    256,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
        with scope('conv2'):
            data = relu(
                conv(
                    data, 256, 1, 1, groups=1, padding=0,
                    param_attr=param_attr))
        with scope('conv3'):
            data = relu(
                conv(
                    data, 256, 1, 1, groups=1, padding=0,
                    param_attr=param_attr))
        with scope('conv4'):
            data = conv(
                data,
                num_classes,
                1,
                1,
                groups=1,
                padding=0,
                param_attr=param_attr)
        return data


def get_points(prediction,
               N,
               k=3,
               beta=0.75,
               label=None,
               phase=ModelPhase.TRAIN):
    '''
    根据分割部分预测结果的不确定性选取渲染的点
    :param prediction: 分割预测结果，已经经过插值处理
    :param N: 渲染的点数
    :param k: 过采样的倍数
    :param beta: 重要点的比例
    :param label: 标注图
    :return: 返回待渲染的点
    '''
    prediction_softmax = fluid.layers.softmax(prediction, axis=1)
    prediction_softmax = fluid.layers.transpose(prediction_softmax,
                                                [0, 2, 3, 1])
    top2, _ = fluid.layers.topk(prediction_softmax, k=2)
    uncertain_features = fluid.layers.abs(top2[:, :, :, 0] - top2[:, :, :, 1])
    fea_shape = uncertain_features.shape
    bs = cfg.batch_size_per_dev
    num_fea_points = fea_shape[-1] * fea_shape[-2]
    uncertain_features = fluid.layers.reshape(
        uncertain_features, shape=(bs, num_fea_points))
    if not ModelPhase.is_train(phase):
        _, index = fluid.layers.argsort(uncertain_features, axis=-1)
        uncertain_points = index[:, :N]
        return uncertain_points
    else:
        # 获取过采样点, 并排除ignore
        label = fluid.layers.reshape(label, shape=(bs, num_fea_points))
        ignore_mask = label == 255
        ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
        rand_tensor = fluid.layers.uniform_random(
            shape=(bs, num_fea_points), min=0, max=1)
        rand_tensor = rand_tensor - ignore_mask
        _, points = fluid.layers.topk(rand_tensor, k=k * N)

        # 获取重要点
        important_points = []
        for i in range(bs):
            points_i = points[i]
            points_i = fluid.layers.unsqueeze(points_i, axes=[-1])
            fea_points_i = fluid.layers.gather_nd(uncertain_features[i],
                                                  points_i)
            _, importance_index = fluid.layers.topk(
                -1 * fea_points_i, k=int(beta * N))
            importance_index = fluid.layers.unsqueeze(
                importance_index, axes=[-1])
            important_points_i = fluid.layers.gather_nd(points[i],
                                                        importance_index)
            important_points_i = fluid.layers.unsqueeze(
                important_points_i, axes=[0])
            important_points.append(important_points_i)
        important_points = fluid.layers.concat(important_points, axis=0)

        # 随机点获取（1-beta*N)
        rand_tensor = fluid.layers.uniform_random(
            shape=(bs, num_fea_points), min=0, max=1)
        rand_tensor = rand_tensor - ignore_mask
        _, rand_points = fluid.layers.topk(rand_tensor, k=N - int(beta * N))

        uncertain_points = fluid.layers.concat([important_points, rand_points],
                                               axis=-1)
        return uncertain_points


def get_point_wise_features(fine_features, prediction, points):
    '''获取point wise features，shape为（bs, c, N, 1)'''
    bs = cfg.batch_size_per_dev
    c_fine, h, w = fine_features.shape[1:]
    num_fea_points = h * w
    c_pred = prediction.shape[1]
    fine_features = fluid.layers.transpose(fine_features, [0, 2, 3, 1])
    prediction = fluid.layers.transpose(prediction, [0, 2, 3, 1])
    fine_features = fluid.layers.reshape(fine_features,
                                         (bs, num_fea_points, c_fine))
    prediction = fluid.layers.reshape(prediction, (bs, num_fea_points, c_pred))

    # pwf为point wise features的缩写
    pwf_fine = []
    pwf_pred = []
    for i in range(bs):
        points_i = points[i]
        points_i = fluid.layers.unsqueeze(points_i, axes=[-1])
        pwf_fine_i = fluid.layers.gather_nd(fine_features[i], points_i)
        pwf_pred_i = fluid.layers.gather_nd(prediction[i], points_i)
        pwf_fine_i = fluid.layers.unsqueeze(pwf_fine_i, axes=0)
        pwf_pred_i = fluid.layers.unsqueeze(pwf_pred_i, axes=0)
        pwf_fine.append(pwf_fine_i)
        pwf_pred.append(pwf_pred_i)
        points_i.stop_gradient = True
    pwf_fine = fluid.layers.concat(pwf_fine, axis=0)
    pwf_pred = fluid.layers.concat(pwf_pred, axis=0)
    pwf = fluid.layers.concat([pwf_fine, pwf_pred], axis=-1)
    pwf = fluid.layers.transpose(pwf, [0, 2, 1])
    pwf = fluid.layers.unsqueeze(pwf, axes=-1)

    return pwf


def render(fine_feature,
           coarse_pred,
           size,
           N,
           num_classes,
           label=None,
           phase=ModelPhase.TRAIN):
    # 插值到需要渲染的大小
    interpolation_coarse_prediction = fluid.layers.resize_bilinear(
        coarse_pred, size)
    interpolation_fine_feature = fluid.layers.resize_bilinear(
        fine_feature, size)
    points = get_points(
        interpolation_coarse_prediction,
        N=N,
        k=cfg.MODEL.POINTREND.K,
        beta=cfg.MODEL.POINTREND.BETA,
        label=label,
        phase=phase)
    point_wise_features = get_point_wise_features(
        interpolation_fine_feature, interpolation_coarse_prediction, points)
    render_mlp = mlp(point_wise_features, num_classes)
    if ModelPhase.is_train(phase):
        return interpolation_coarse_prediction, render_mlp, points
    else:
        # 渲染点概率替换
        bs = cfg.batch_size_per_dev
        c, h, w = interpolation_coarse_prediction.shape[1:]
        interpolation_coarse_prediction = fluid.layers.transpose(
            interpolation_coarse_prediction, [0, 2, 3, 1])
        interpolation_coarse_prediction = fluid.layers.reshape(
            interpolation_coarse_prediction, (bs, h * w, c))

        render_mlp = fluid.layers.squeeze(render_mlp, axes=[-1])
        render_mlp = fluid.layers.transpose(render_mlp, [0, 2, 1])
        interpolation_coarse_prediction_mlp = []
        for i in range(bs):
            interpolation_coarse_prediction_i = interpolation_coarse_prediction[
                i]
            points_i = points[i]
            render_mlp_i = render_mlp[i]
            points_i = fluid.layers.unsqueeze(points_i, axes=[-1])
            # 渲染点置零
            mask = fluid.layers.ones_like(interpolation_coarse_prediction_i)
            updates_mask = 0 - fluid.layers.ones(shape=(N, c), dtype='float32')
            mask = fluid.layers.scatter_nd_add(mask, points_i, updates_mask)
            # 渲染点替换
            interpolation_coarse_prediction_i = fluid.layers.elementwise_mul(
                interpolation_coarse_prediction_i, mask)
            interpolation_coarse_prediction_i = fluid.layers.scatter_nd_add(
                interpolation_coarse_prediction_i, points_i, render_mlp_i)
            interpolation_coarse_prediction_i = fluid.layers.unsqueeze(
                interpolation_coarse_prediction_i, axes=0)
            interpolation_coarse_prediction_mlp.append(
                interpolation_coarse_prediction_i)
        interpolation_coarse_prediction_mlp = fluid.layers.concat(
            interpolation_coarse_prediction_mlp, axis=0)
        interpolation_coarse_prediction_mlp = fluid.layers.reshape(
            interpolation_coarse_prediction_mlp, (bs, h, w, c))
        interpolation_coarse_prediction_mlp = fluid.layers.transpose(
            interpolation_coarse_prediction_mlp, [0, 3, 1, 2])
        return interpolation_coarse_prediction_mlp, render_mlp, points


def pointrend(coarse_pred,
              fine_feature,
              num_classes,
              input_size,
              label=None,
              phase=ModelPhase.TRAIN):
    coarse_size = coarse_pred.shape
    N = coarse_size[-1] * coarse_size[-2]
    # 计算渲染的次数
    if ModelPhase.is_train(phase):
        coarse_pred = fluid.layers.resize_bilinear(coarse_pred, input_size[-2:])
        outs = [(coarse_pred, )]
        _, render_mlp, points = render(
            fine_feature,
            coarse_pred,
            input_size[-2:],
            N=N,
            num_classes=num_classes,
            label=label,
            phase=phase)
        outs.append((render_mlp, points))
        return outs

    else:
        num_render = int(np.log2(input_size[-1] / coarse_size[-1]) + 0.5)
        for k in range(num_render - 1):
            size = [2**(k + 1) * i for i in coarse_size[-2:]]
            coarse_pred, _, _ = render(
                fine_feature,
                coarse_pred,
                size,
                N=N,
                num_classes=num_classes,
                phase=phase)
        prediction, _, _ = render(
            fine_feature,
            coarse_pred,
            input_size[-2:],
            N=N,
            num_classes=num_classes,
            phase=phase)
        return prediction
