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

import os

import paddle
from urllib.parse import urlparse
from paddleseg.utils import logger, download_pretrained_model


def get_files(root_path):
    res = []
    for root, dirs, files in os.walk(root_path, followlinks=True):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg', 'JPG')):
                res.append(os.path.join(root, f))
    return res


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        image_dir = None
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        raise RuntimeError(
                            'There should be only one image path per line in `image_path` file. Wrong line: {}'
                            .format(line))
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
        image_list.sort()
    else:
        raise FileNotFoundError(
            '`image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `image_path`')

    return image_list, image_dir


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logger.info('Loading pretrained model from {}'.format(pretrained_model))

        if urlparse(pretrained_model).netloc:
            pretrained_model = download_pretrained_model(pretrained_model)

        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    # When the input is more than 3 channels such as trimap-based method, padding zeros to load.
                    para_shape = list(para_state_dict[k].shape)
                    model_shape = list(model_state_dict[k].shape)
                    if 'weight' in k \
                        and len(para_shape) > 3 \
                        and len(para_shape) > 3 \
                        and para_shape[1] < model_shape[1] \
                        and para_shape[0] == model_shape[0] \
                        and para_shape[2] == model_shape[2] \
                        and para_shape[3] == model_shape[3]:
                        zeros_pad = paddle.zeros(
                            (para_shape[0], model_shape[1] - para_shape[1],
                             para_shape[2], para_shape[3]))
                        para_state_dict[k] = paddle.concat(
                            [para_state_dict[k], zeros_pad], axis=1)
                        model_state_dict[k] = para_state_dict[k]
                        num_params_loaded += 1
                    else:
                        logger.warning(
                            "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                            .format(k, para_state_dict[k].shape,
                                    model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrained_model))
    else:
        logger.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))
