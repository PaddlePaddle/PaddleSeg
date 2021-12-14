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
import os
import cv2
import imageio
import numpy as np

from paddleseg.utils import logger
import PIL


def load_ema_model(model, resume_model):
    if resume_model is not None:
        logger.info('Load ema model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def save_edge(edges_src, name):
    tmp = edges_src.detach().clone().squeeze().numpy()
    tmp[tmp == 1] == 255
    imageio.imwrite('edge_pics/edge_{}.png'.format(name), tmp)


def get_color_map_list(num_classes):
    colormap = [
        128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
        153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152,
        0, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0,
        80, 100, 0, 0, 230, 119, 11, 32
    ] + [0, 0, 0] * (256 - num_classes)

    return colormap


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.

    Args:
        pred (numpy.ndarray): the origin predicted image, H*W .
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.

    Returns:
        (numpy.ndarray): the pseduo image.
    """
    if len(pred.shape) > 2:
        pred = np.squeeze(pred)

    pred_mask = PIL.Image.fromarray(pred.astype(np.uint8), mode='P')
    color_map = get_color_map_list(19)
    pred_mask.putpalette(color_map)
    return pred_mask


def save_imgs(results, imgs, save_dir='.'):
    for i in range(results.shape[0]):
        result = get_pseudo_color_map(results[i])
        basename = imgs[i] + 'val'
        basename = f'{basename}.png'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        result.save(os.path.join(save_dir, basename))
