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

import sys, os
sys.path.append(os.path.abspath('../../'))
import paddle.fluid as fluid
from utils.config import cfg
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, avg_pool, conv, bn_relu, relu
from models.libs.model_libs import separate_conv
from models.model_builder import ModelPhase
from models.backbone.resnet import ResNet as resnet_backbone
import numpy as np


def resnet(input):
    # PointRend backbone: resnet, 默认resnet101
    # end_points: resnet终止层数
    # decode_point: backbone引出分支所在层数
    scale = cfg.MODEL.POINTREND.DEPTH_MULTIPLIER
    layers = cfg.MODEL.POINTREND.LAYERS
    model = resnet_backbone(scale=scale, layers=layers, stem='pointrend')
    end_points = 100
    decode_point = 22
    data, decode_shortcuts = model.net(
        input, end_points=end_points, decode_points=decode_point)
    return data, decode_shortcuts[decode_point]


def encoder(input):
    # 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
    # ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
    # OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
    # aspp_ratios：ASPP模块空洞卷积的采样率

    if cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 16:
        aspp_ratios = [6, 12, 18]
    elif cfg.MODEL.DEEPLAB.OUTPUT_STRIDE == 8:
        aspp_ratios = [12, 24, 36]
    else:
        raise Exception("deeplab only support stride 8 or 16")

    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('encoder'):
        channel = 256
        with scope("image_pool"):
            image_avg = fluid.layers.reduce_mean(input, [2, 3], keep_dim=True)
            image_avg = bn_relu(
                conv(
                    image_avg,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
            image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])

        with scope("aspp0"):
            aspp0 = bn_relu(
                conv(
                    input,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
        with scope("aspp1"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp1 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[0], act=relu)
            else:
                aspp1 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[0],
                        padding=aspp_ratios[0],
                        param_attr=param_attr))
        with scope("aspp2"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp2 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[1], act=relu)
            else:
                aspp2 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[1],
                        padding=aspp_ratios[1],
                        param_attr=param_attr))
        with scope("aspp3"):
            if cfg.MODEL.DEEPLAB.ASPP_WITH_SEP_CONV:
                aspp3 = separate_conv(
                    input, channel, 1, 3, dilation=aspp_ratios[2], act=relu)
            else:
                aspp3 = bn_relu(
                    conv(
                        input,
                        channel,
                        stride=1,
                        filter_size=3,
                        dilation=aspp_ratios[2],
                        padding=aspp_ratios[2],
                        param_attr=param_attr))
        with scope("concat"):
            data = fluid.layers.concat([image_avg, aspp0, aspp1, aspp2, aspp3],
                                       axis=1)
            data = bn_relu(
                conv(
                    data,
                    channel,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))
            data = fluid.layers.dropout(data, 0.9)
        return data


def deeplabv3(img, num_classes):
    data, decode_shortcut = resnet(img)
    data = encoder(data)

    # 根据类别数设置最后一个卷积层输出，并resize到图片原始尺寸
    param_attr = fluid.ParamAttr(
        name=name_scope + 'weights',
        regularizer=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0),
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
    with scope('logit'):
        with fluid.name_scope('last_conv'):
            logit = conv(
                data,
                num_classes,
                1,
                stride=1,
                padding=0,
                bias_attr=True,
                param_attr=param_attr)
        # logit = fluid.layers.resize_bilinear(logit, img.shape[2:])

    return logit, decode_shortcut


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
    根据depelabv3预测结果的不确定性选取渲染的点
    :param prediction: depelabv3预测结果，已经经过插值处理
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
    bs = cfg.BATCH_SIZE
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
            # points_i = fluid.layers.Print(points_i)
            # fea_points_i = fluid.layers.Print(fea_points_i)
            # important_points_i = fluid.layers.Print(important_points_i)
        important_points = fluid.layers.concat(important_points, axis=0)
        important_points = fluid.layers.Print(important_points)

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
    bs = cfg.BATCH_SIZE
    c_fine, h, w = fine_features.shape[1:]
    num_fea_points = h * w
    c_pred = prediction.shape[1]
    fine_features = fluid.layers.transpose(fine_features, [0, 2, 3, 1])
    prediction = fluid.layers.transpose(prediction, [0, 2, 3, 1])
    fine_features = fluid.layers.reshape(fine_features,
                                         (bs, num_fea_points, c_fine))
    prediction = fluid.layers.reshape(prediction, (bs, num_fea_points, c_pred))

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
    inter_coarse_prediction = fluid.layers.resize_bilinear(coarse_pred, size)
    inter_fine_feature = fluid.layers.resize_bilinear(fine_feature, size)
    print(inter_coarse_prediction.shape)
    print(inter_fine_feature.shape)
    if label is not None:
        label = fluid.layers.resize_nearest(label, size)
    points = get_points(
        inter_coarse_prediction,
        N=N,
        k=cfg.MODEL.POINTREND.K,
        beta=cfg.MODEL.POINTREND.BETA,
        label=label,
        phase=phase)
    print('points\n', points.shape)
    point_wise_features = get_point_wise_features(
        inter_fine_feature, inter_coarse_prediction, points)
    render_mlp = mlp(point_wise_features, num_classes)
    if ModelPhase.is_train(phase):
        return inter_coarse_prediction, render_mlp, points
    else:
        # 渲染点概率替换
        bs = cfg.BATCH_SIZE
        c, h, w = inter_coarse_prediction.shape[1:]
        inter_coarse_prediction = fluid.layers.transpose(
            inter_coarse_prediction, [0, 2, 3, 1])
        inter_coarse_prediction = fluid.layers.reshape(inter_coarse_prediction,
                                                       (bs, h * w, c))

        render_mlp = fluid.layers.squeeze(render_mlp, axes=[-1])
        render_mlp = fluid.layers.transpose(render_mlp, [0, 2, 1])
        inter_coarse_prediction_mlp = []
        for i in range(bs):
            inter_coarse_prediction_i = inter_coarse_prediction[i]
            points_i = points[i]
            render_mlp_i = render_mlp[i]
            points_i = fluid.layers.unsqueeze(points_i, axes=[-1])
            # 渲染点置零
            mask = fluid.layers.ones_like(inter_coarse_prediction_i)
            updates_mask = -fluid.layers.ones(shape=(N, c), dtype='float32')
            print(mask.shape)
            print(points_i.shape)
            print(updates_mask.shape)
            mask = fluid.layers.scatter_nd_add(mask, points_i, updates_mask)
            # 渲染点替换
            inter_coarse_prediction_i = fluid.layers.elementwise_mul(
                inter_coarse_prediction_i, mask)
            inter_coarse_prediction_i = fluid.layers.scatter_nd_add(
                inter_coarse_prediction_i, points_i, render_mlp_i)
            inter_coarse_prediction_i = fluid.layers.unsqueeze(
                inter_coarse_prediction_i, axes=0)
            inter_coarse_prediction_mlp.append(inter_coarse_prediction_i)
        inter_coarse_prediction_mlp = fluid.layers.concat(
            inter_coarse_prediction_mlp, axis=0)
        inter_coarse_prediction_mlp = fluid.layers.reshape(
            inter_coarse_prediction_mlp, (bs, h, w, c))
        inter_coarse_prediction_mlp = fluid.layers.transpose(
            inter_coarse_prediction_mlp, [0, 3, 1, 2])
        return inter_coarse_prediction_mlp, render_mlp, points


def pointrend(img, num_classes, label=None, phase=ModelPhase.TRAIN):
    coarse_pred, fine_feature = deeplabv3(img, num_classes)
    input_size = img.shape
    coarse_size = coarse_pred.shape
    N = coarse_size[-1] * coarse_size[-2]
    # 计算渲染的次数
    if ModelPhase.is_train(phase):
        outs = [(coarse_pred, )]
        _, render_mlp, points = render(
            fine_feature,
            coarse_pred,
            input_size[-2:],
            N=N,
            num_classes=num_classes,
            label=label,
            phase=phase)
        outs.append((render_mlp, points, input_size[-2:]))
        return outs

    else:
        num_render = int(np.log2(input_size[-1] / coarse_size[-1]) + 0.5)
        print(input_size, coarse_size)
        print(num_render)
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


if __name__ == '__main__':
    import os
    os.environ['GLOG_vmodule'] = '4'
    os.environ['GLOG_logtostderr'] = '1'

    image_shape = [2, 3, 769, 769]
    label_shape = [2, 1, 769, 769]
    cfg.BATCH_SIZE = 2
    image = fluid.layers.data(
        name='image',
        shape=image_shape,
        dtype='float32',
        append_batch_size=False)
    label = fluid.layers.data(
        name='label', shape=label_shape, dtype='int64', append_batch_size=False)
    # points = get_points(image, N=3, phase=ModelPhase.TRAIN, label=label)
    # pwf = get_point_wise_features(image, image, points)
    out = pointrend(image, 2, label, ModelPhase.EVAL)
    # for i in out:
    #     for j in i:
    #         print(j)

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    places = fluid.cpu_places()
    place = places[0]

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    print('out', out.shape)
    a = np.random.randint(0, 256, size=image_shape).astype("float32")
    b = np.random.randint(0, 2, size=label_shape).astype("int64")
    out = exe.run(
        program=train_prog,
        feed={
            image.name: a,
            label.name: b
        },
        fetch_list=[out.name])
    print(out[0])
    # print('input', a.transpose([0, 2, 3, 1]))
    # print('uncertaion_points\n', out[0])
    # print('pwf\n', out[1])
