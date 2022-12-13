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

from paddlepanseg.transforms.functional import id2rgb
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


def get_save_name(im_path, im_dir, ext='.png'):
    """Get the saved name"""
    if im_dir is not None:
        im_file = im_path.replace(im_dir, '')
    else:
        im_file = osp.basename(im_path)
    if im_file[0] == '/':
        im_file = im_file[1:]
    if ext is not None:
        im_file = osp.splitext(im_file)[0] + ext
    return im_file


def predict(model,
            model_path,
            transforms,
            postprocessor,
            image_list,
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
            ps = id2rgb(pp_out['pan_pred'][0, 0].numpy())

            im_file = get_save_name(im_path, image_dir)
            Image.fromarray(ps).convert('RGB').save(osp.join(save_dir, im_file))

            progbar_pred.update(i + 1)
