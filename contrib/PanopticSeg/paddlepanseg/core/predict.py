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
import os.path as osp
import math

import numpy as np
import paddle
import paddleseg
from paddleseg.utils import logger, progbar
from PIL import Image

from paddlepanseg.utils.visualize import visualize_instance, visualize_semantic, visualize_panoptic
from paddlepanseg.cvlibs import build_info_dict
from . import infer


def mkdir(path):
    sub_dir = osp.dirname(path)
    if not osp.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """Split the list `arr` into `m` pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def get_save_name(im_path, im_dir, suffix='.png'):
    """Get the saved name"""
    if im_dir is not None:
        im_file = im_path.replace(im_dir, '')
    else:
        im_file = osp.basename(im_path)
    if im_file[0] == '/':
        im_file = im_file[1:]
    if suffix is not None:
        im_file = osp.splitext(im_file)[0] + suffix
    return im_file


def predict(model,
            model_path,
            transforms,
            postprocessor,
            image_list,
            label_divisor,
            ignore_index,
            colormap=None,
            image_dir=None,
            save_dir='output'):
    """
    Predict the panoptic segmentation results given the input of `image_list`.

    Args:
        model (nn.Layer): A panoptic segmentation model.
        model_path (str): The path of the pretrained model.
        transforms (paddleseg.transforms.Compose): The pipeline for data preprocessing of the input image.
        postprocessor (paddlepanseg.postprocessors.Postprocessor): Used to postprocess model output.
        image_list (list): A list of image path to be predicted.
        label_divisor (int): Used for conversion between semantic IDs and panoptic IDs.
        ignore_index (int): The class ID to be ignored. Default: 255.
        color_map (list, optional): The colormap used for visualization. Default: None.
        image_dir (str, optional): The root directory of the images to predict. Default: None.
        save_dir (str, optional): The directory for saving the predicted results. Default: 'output'.
    """
    paddleseg.utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            data = build_info_dict(
                _type_='sample', img=im_path, img_path=im_path)
            data = transforms(data)
            # XXX: For efficiency only convert img to tensor.
            data['img'] = paddle.to_tensor(data['img']).unsqueeze(0)

            pp_out = infer.inference(
                model=model, data=data, postprocessor=postprocessor)
            # NOTE: We do not perform id conversion before visualization
            pan_pred = pp_out['pan_pred'][0, 0].numpy()
            sem_pred = pp_out['sem_pred'][0, 0].numpy()
            sem_vis = visualize_semantic(sem_pred, colormap)
            ins_vis = visualize_instance(pan_pred, ignore_ins_id=0)
            pan_vis = visualize_panoptic(
                pan_pred,
                label_divisor=label_divisor,
                colormap=colormap,
                ignore_index=ignore_index)

            Image.fromarray(sem_vis).convert('RGB').save(
                osp.join(save_dir,
                         get_save_name(im_path, image_dir, '_sem.png')))
            Image.fromarray(ins_vis).convert('RGB').save(
                osp.join(save_dir,
                         get_save_name(im_path, image_dir, '_ins.png')))
            Image.fromarray(pan_vis).convert('RGB').save(
                osp.join(save_dir,
                         get_save_name(im_path, image_dir, '_pan.png')))

            progbar_pred.update(i + 1)
