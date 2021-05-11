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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid

from utils.config import cfg
from pdseg.models.backbone.vgg import VGGNet as vgg_backbone
from pdseg.models.libs.model_libs import bn, relu
from pdseg.models.libs.model_libs import conv, max_pool, deconv
from pdseg.models.libs.model_libs import scope

# from models.backbone.vgg import VGGNet as vgg_backbone

# Bottleneck type
REGULAR = 1
DOWNSAMPLING = 2
UPSAMPLING = 3
DILATED = 4
ASYMMETRIC = 5


def prelu(x, decoder=False):
    # If decoder, then perform relu else perform prelu
    if decoder:
        return fluid.layers.relu(x)
    return fluid.layers.prelu(x, 'channel')


def initial_block(inputs, name_scope='initial_block'):
    '''
    The initial block for ENet has 2 branches: The convolution branch and MaxPool branch.
    The conv branch has 13 filters, while the maxpool branch gives 3 channels corresponding to the RGB channels.
    Both output layers are then concatenated to give an output of 16 channels.

    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :return net_concatenated(Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''
    # Convolutional branch
    with scope(name_scope):
        net_conv = conv(inputs, 13, 3, stride=2, padding=1)
        net_conv = bn(net_conv)
        net_conv = fluid.layers.prelu(net_conv, 'channel')

        # Max pool branch
        net_pool = max_pool(inputs, [2, 2], stride=2, padding='SAME')

        # Concatenated output - does it matter max pool comes first or conv comes first? probably not.
        net_concatenated = fluid.layers.concat([net_conv, net_pool], axis=1)
    return net_concatenated


def bottleneck(inputs,
               output_depth,
               filter_size,
               regularizer_prob,
               projection_ratio=4,
               type=REGULAR,
               seed=0,
               output_shape=None,
               dilation_rate=None,
               decoder=False,
               name_scope='bottleneck'):
    # Calculate the depth reduction based on the projection ratio used in 1x1 convolution.
    reduced_depth = int(inputs.shape[1] / projection_ratio)

    # DOWNSAMPLING BOTTLENECK
    if type == DOWNSAMPLING:
        # =============MAIN BRANCH=============
        # Just perform a max pooling
        with scope('down_sample'):
            inputs_shape = inputs.shape
            with scope('main_max_pool'):
                net_main = fluid.layers.conv2d(
                    inputs,
                    inputs_shape[1],
                    filter_size=3,
                    stride=2,
                    padding='SAME')

            # First get the difference in depth to pad, then pad with zeros only on the last dimension.
            depth_to_pad = abs(inputs_shape[1] - output_depth)
            paddings = [0, 0, 0, depth_to_pad, 0, 0, 0, 0]
            with scope('main_padding'):
                net_main = fluid.layers.pad(net_main, paddings=paddings)

            with scope('block1'):
                net = conv(
                    inputs, reduced_depth, [2, 2], stride=2, padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            with scope('block2'):
                net = conv(
                    net,
                    reduced_depth, [filter_size, filter_size],
                    padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            with scope('block3'):
                net = conv(net, output_depth, [1, 1], padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Regularizer
            net = fluid.layers.dropout(net, regularizer_prob, seed=seed)

            # Finally, combine the two branches together via an element-wise addition
            net = fluid.layers.elementwise_add(net, net_main)
            net = prelu(net, decoder=decoder)

        return net, inputs_shape

    # DILATION CONVOLUTION BOTTLENECK
    # Everything is the same as a regular bottleneck except for the dilation rate argument
    elif type == DILATED:
        # Check if dilation rate is given
        if not dilation_rate:
            raise ValueError('Dilation rate is not given.')

        with scope('dilated'):
            # Save the main branch for addition later
            net_main = inputs

            # First projection with 1x1 kernel (dimensionality reduction)
            with scope('block1'):
                net = conv(inputs, reduced_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Second conv block --- apply dilated convolution here
            with scope('block2'):
                net = conv(
                    net,
                    reduced_depth,
                    filter_size,
                    padding='SAME',
                    dilation=dilation_rate)
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Final projection with 1x1 kernel (Expansion)
            with scope('block3'):
                net = conv(net, output_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Regularizer
            net = fluid.layers.dropout(net, regularizer_prob, seed=seed)
            net = prelu(net, decoder=decoder)

            # Add the main branch
            net = fluid.layers.elementwise_add(net_main, net)
            net = prelu(net, decoder=decoder)

        return net

    # ASYMMETRIC CONVOLUTION BOTTLENECK
    # Everything is the same as a regular bottleneck except for a [5,5] kernel decomposed into two [5,1] then [1,5]
    elif type == ASYMMETRIC:
        # Save the main branch for addition later
        with scope('asymmetric'):
            net_main = inputs
            # First projection with 1x1 kernel (dimensionality reduction)
            with scope('block1'):
                net = conv(inputs, reduced_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Second conv block --- apply asymmetric conv here
            with scope('block2'):
                with scope('asymmetric_conv2a'):
                    net = conv(
                        net, reduced_depth, [filter_size, 1], padding='same')
                with scope('asymmetric_conv2b'):
                    net = conv(
                        net, reduced_depth, [1, filter_size], padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Final projection with 1x1 kernel
            with scope('block3'):
                net = conv(net, output_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Regularizer
            net = fluid.layers.dropout(net, regularizer_prob, seed=seed)
            net = prelu(net, decoder=decoder)

            # Add the main branch
            net = fluid.layers.elementwise_add(net_main, net)
            net = prelu(net, decoder=decoder)

        return net

    # UPSAMPLING BOTTLENECK
    # Everything is the same as a regular one, except convolution becomes transposed.
    elif type == UPSAMPLING:
        # Check if pooling indices is given

        # Check output_shape given or not
        if output_shape is None:
            raise ValueError('Output depth is not given')

        # =======MAIN BRANCH=======
        # Main branch to upsample. output shape must match with the shape of the layer that was pooled initially, in order
        # for the pooling indices to work correctly. However, the initial pooled layer was padded, so need to reduce dimension
        # before unpooling. In the paper, padding is replaced with convolution for this purpose of reducing the depth!
        with scope('upsampling'):
            with scope('unpool'):
                net_unpool = conv(inputs, output_depth, [1, 1])
                net_unpool = bn(net_unpool)
                net_unpool = fluid.layers.resize_bilinear(
                    net_unpool, out_shape=output_shape[2:])

            # First 1x1 projection to reduce depth
            with scope('block1'):
                net = conv(inputs, reduced_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            with scope('block2'):
                net = deconv(
                    net,
                    reduced_depth,
                    filter_size=filter_size,
                    stride=2,
                    padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Final projection with 1x1 kernel
            with scope('block3'):
                net = conv(net, output_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Regularizer
            net = fluid.layers.dropout(net, regularizer_prob, seed=seed)
            net = prelu(net, decoder=decoder)

            # Finally, add the unpooling layer and the sub branch together
            net = fluid.layers.elementwise_add(net, net_unpool)
            net = prelu(net, decoder=decoder)

        return net

    # REGULAR BOTTLENECK
    else:
        with scope('regular'):
            net_main = inputs

            # First projection with 1x1 kernel
            with scope('block1'):
                net = conv(inputs, reduced_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Second conv block
            with scope('block2'):
                net = conv(
                    net,
                    reduced_depth, [filter_size, filter_size],
                    padding='same')
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Final projection with 1x1 kernel
            with scope('block3'):
                net = conv(net, output_depth, [1, 1])
                net = bn(net)
                net = prelu(net, decoder=decoder)

            # Regularizer
            net = fluid.layers.dropout(net, regularizer_prob, seed=seed)
            net = prelu(net, decoder=decoder)

            # Add the main branch
            net = fluid.layers.elementwise_add(net_main, net)
            net = prelu(net, decoder=decoder)

        return net


def ENet_stage1(inputs, name_scope='stage1_block'):
    with scope(name_scope):
        with scope('bottleneck1_0'):
            net, inputs_shape_1 \
                = bottleneck(inputs, output_depth=64, filter_size=3, regularizer_prob=0.01, type=DOWNSAMPLING,
                             name_scope='bottleneck1_0')
        with scope('bottleneck1_1'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.01,
                name_scope='bottleneck1_1')
        with scope('bottleneck1_2'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.01,
                name_scope='bottleneck1_2')
        with scope('bottleneck1_3'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.01,
                name_scope='bottleneck1_3')
        with scope('bottleneck1_4'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.01,
                name_scope='bottleneck1_4')
    return net, inputs_shape_1


def ENet_stage2(inputs, name_scope='stage2_block'):
    with scope(name_scope):
        net, inputs_shape_2 \
            = bottleneck(inputs, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DOWNSAMPLING,
                         name_scope='bottleneck2_0')
        for i in range(2):
            with scope('bottleneck2_{}'.format(str(4 * i + 1))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    name_scope='bottleneck2_{}'.format(str(4 * i + 1)))
            with scope('bottleneck2_{}'.format(str(4 * i + 2))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    type=DILATED,
                    dilation_rate=(2**(2 * i + 1)),
                    name_scope='bottleneck2_{}'.format(str(4 * i + 2)))
            with scope('bottleneck2_{}'.format(str(4 * i + 3))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=5,
                    regularizer_prob=0.1,
                    type=ASYMMETRIC,
                    name_scope='bottleneck2_{}'.format(str(4 * i + 3)))
            with scope('bottleneck2_{}'.format(str(4 * i + 4))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    type=DILATED,
                    dilation_rate=(2**(2 * i + 2)),
                    name_scope='bottleneck2_{}'.format(str(4 * i + 4)))
    return net, inputs_shape_2


def ENet_stage3(inputs, name_scope='stage3_block'):
    with scope(name_scope):
        for i in range(2):
            with scope('bottleneck3_{}'.format(str(4 * i + 0))):
                net = bottleneck(
                    inputs,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    name_scope='bottleneck3_{}'.format(str(4 * i + 0)))
            with scope('bottleneck3_{}'.format(str(4 * i + 1))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    type=DILATED,
                    dilation_rate=(2**(2 * i + 1)),
                    name_scope='bottleneck3_{}'.format(str(4 * i + 1)))
            with scope('bottleneck3_{}'.format(str(4 * i + 2))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=5,
                    regularizer_prob=0.1,
                    type=ASYMMETRIC,
                    name_scope='bottleneck3_{}'.format(str(4 * i + 2)))
            with scope('bottleneck3_{}'.format(str(4 * i + 3))):
                net = bottleneck(
                    net,
                    output_depth=128,
                    filter_size=3,
                    regularizer_prob=0.1,
                    type=DILATED,
                    dilation_rate=(2**(2 * i + 2)),
                    name_scope='bottleneck3_{}'.format(str(4 * i + 3)))
    return net


def ENet_stage4(inputs,
                inputs_shape,
                connect_tensor,
                skip_connections=True,
                name_scope='stage4_block'):
    with scope(name_scope):
        with scope('bottleneck4_0'):
            net = bottleneck(
                inputs,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.1,
                type=UPSAMPLING,
                decoder=True,
                output_shape=inputs_shape,
                name_scope='bottleneck4_0')

        if skip_connections:
            net = fluid.layers.elementwise_add(net, connect_tensor)
        with scope('bottleneck4_1'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.1,
                decoder=True,
                name_scope='bottleneck4_1')
        with scope('bottleneck4_2'):
            net = bottleneck(
                net,
                output_depth=64,
                filter_size=3,
                regularizer_prob=0.1,
                decoder=True,
                name_scope='bottleneck4_2')

    return net


def ENet_stage5(inputs,
                inputs_shape,
                connect_tensor,
                skip_connections=True,
                name_scope='stage5_block'):
    with scope(name_scope):
        net = bottleneck(
            inputs,
            output_depth=16,
            filter_size=3,
            regularizer_prob=0.1,
            type=UPSAMPLING,
            decoder=True,
            output_shape=inputs_shape,
            name_scope='bottleneck5_0')

        if skip_connections:
            net = fluid.layers.elementwise_add(net, connect_tensor)
        with scope('bottleneck5_1'):
            net = bottleneck(
                net,
                output_depth=16,
                filter_size=3,
                regularizer_prob=0.1,
                decoder=True,
                name_scope='bottleneck5_1')
    return net


def decoder(input, num_classes):
    if 'enet' in cfg.MODEL.LANENET.BACKBONE:
        # Segmentation branch
        with scope('LaneNetSeg'):
            initial, stage1, stage2, inputs_shape_1, inputs_shape_2 = input
            segStage3 = ENet_stage3(stage2)
            segStage4 = ENet_stage4(segStage3, inputs_shape_2, stage1)
            segStage5 = ENet_stage5(segStage4, inputs_shape_1, initial)
            segLogits = deconv(
                segStage5, num_classes, filter_size=2, stride=2, padding='SAME')

        # Embedding branch
        with scope('LaneNetEm'):
            emStage3 = ENet_stage3(stage2)
            emStage4 = ENet_stage4(emStage3, inputs_shape_2, stage1)
            emStage5 = ENet_stage5(emStage4, inputs_shape_1, initial)
            emLogits = deconv(
                emStage5, 4, filter_size=2, stride=2, padding='SAME')

    elif 'vgg' in cfg.MODEL.LANENET.BACKBONE:
        encoder_list = ['pool5', 'pool4', 'pool3']
        # score stage
        input_tensor = input[encoder_list[0]]
        with scope('score_origin'):
            score = conv(input_tensor, 64, 1)
        encoder_list = encoder_list[1:]
        for i in range(len(encoder_list)):
            with scope('deconv_{:d}'.format(i + 1)):
                deconv_out = deconv(
                    score, 64, filter_size=4, stride=2, padding='SAME')
            input_tensor = input[encoder_list[i]]
            with scope('score_{:d}'.format(i + 1)):
                score = conv(input_tensor, 64, 1)
            score = fluid.layers.elementwise_add(deconv_out, score)

        with scope('deconv_final'):
            emLogits = deconv(
                score, 64, filter_size=16, stride=8, padding='SAME')
        with scope('score_final'):
            segLogits = conv(emLogits, num_classes, 1)
        emLogits = relu(conv(emLogits, 4, 1))
    return segLogits, emLogits


def encoder(input):
    if 'vgg' in cfg.MODEL.LANENET.BACKBONE:
        model = vgg_backbone(layers=16)
        # output = model.net(input)

        _, encode_feature_dict = model.net(
            input, end_points=13, decode_points=[7, 10, 13])
        output = {}
        output['pool3'] = encode_feature_dict[7]
        output['pool4'] = encode_feature_dict[10]
        output['pool5'] = encode_feature_dict[13]
    elif 'enet' in cfg.MODEL.LANENET.BACKBONE:
        with scope('LaneNetBase'):
            initial = initial_block(input)
            stage1, inputs_shape_1 = ENet_stage1(initial)
            stage2, inputs_shape_2 = ENet_stage2(stage1)
            output = (initial, stage1, stage2, inputs_shape_1, inputs_shape_2)
    else:
        raise Exception(
            "LaneNet expect enet and vgg backbone, but received {}".format(
                cfg.MODEL.LANENET.BACKBONE))
    return output


def lanenet(img, num_classes):
    output = encoder(img)
    segLogits, emLogits = decoder(output, num_classes)

    return segLogits, emLogits
