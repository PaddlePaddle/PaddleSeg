# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import math
import cv2
import paddle.fluid as fluid

from . import logging


def seconds_to_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = "{}:{}:{}".format(h, m, s)
    return hms_str


def get_environ_info():
    info = dict()
    info['place'] = 'cpu'
    info['num'] = int(os.environ.get('CPU_NUM', 1))
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) != "":
        if hasattr(fluid.core, 'get_cuda_device_count'):
            gpu_num = 0
            try:
                gpu_num = fluid.core.get_cuda_device_count()
            except:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                pass
            if gpu_num > 0:
                info['place'] = 'cuda'
                info['num'] = fluid.core.get_cuda_device_count()
    return info


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logging.info('Load pretrained model from {}'.format(pretrained_model))
        if os.path.exists(pretrained_model):
            ckpt_path = os.path.join(pretrained_model, 'model')
            try:
                para_state_dict, _ = fluid.load_dygraph(ckpt_path)
            except:
                para_state_dict = fluid.load_program_state(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logging.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(
                        model_state_dict[k].shape):
                    logging.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape,
                                model_state_dict[k].shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logging.info("There are {}/{} varaibles are loaded.".format(
                num_params_loaded, len(model_state_dict)))

        else:
            raise ValueError(
                'The pretrained model directory is not Found: {}'.format(
                    pretrained_model))
    else:
        logging.info('No pretrained model to load, train from scratch')


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logging.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            ckpt_path = os.path.join(resume_model, 'model')
            para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
            model.set_dict(para_state_dict)
            optimizer.set_dict(opti_state_dict)
            epoch = resume_model.split('_')[-1]
            if epoch.isdigit():
                epoch = int(epoch)
            return epoch
        else:
            raise ValueError(
                'The resume model directory is not Found: {}'.format(
                    resume_model))
    else:
        logging.info('No model need to resume')


def visualize(image, result, save_dir=None, weight=0.6):
    """
    Convert segment result to color image, and save added image.
    Args:
        image: the path of origin image
        result: the predict result of image
        save_dir: the directory for saving visual image
        weight: the image weight of visual image, and the result weight is (1 - weight)
    """
    color_map = get_color_map_list(256)
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = color_map[1:]
    return color_map
