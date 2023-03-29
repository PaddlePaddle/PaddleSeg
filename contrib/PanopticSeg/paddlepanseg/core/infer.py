# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F


def reverse_transform_for_single_map(pred, trans_info, mode='nearest'):
    """Recover `pred` to original shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][0], item[1][1]
            ndims = pred.dim()
            if ndims < 4:
                for _ in range(4 - ndims):
                    pred = pred.unsqueeze(0)
            elif ndims > 4:
                raise ValueError
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                # align_corners=False matches the result of OpenCV
                pred = F.interpolate(
                    pred, [h, w], mode=mode, align_corners=False)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(
                    pred, [h, w], mode=mode, align_corners=False)
            if ndims < 4:
                for _ in range(4 - ndims):
                    pred = pred.squeeze(0)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[..., 0:h, 0:w]
        else:
            raise ValueError("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def reverse_transform(net_out, trans_info):
    # XXX: In-place modification
    for key in net_out['map_fields']:
        val = net_out[key]
        if val is None:
            continue
        net_out[key] = reverse_transform_for_single_map(
            val, trans_info=trans_info, mode='bilinear')
    return net_out


def inference(model, data, postprocessor):
    net_out = model(data['img'])

    trans_info = data.get('trans_info', None)
    if trans_info is not None:
        assert len(trans_info) == 1
        trans_info = trans_info[0]
        net_out = reverse_transform(net_out, trans_info)

    pp_out = postprocessor(data, net_out)

    return pp_out
