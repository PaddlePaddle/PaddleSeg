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
import math

import cv2
import numpy as np
import paddle
import paddleseg
from paddleseg.utils import logger, progbar

from core import infer
import utils


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def get_save_name(im_path, im_dir):
    """get the saved name"""
    if im_dir is not None:
        im_file = im_path.replace(im_dir, '')
    else:
        im_file = os.path.basename(im_path)
    if im_file[0] == '/':
        im_file = im_file[1:]
    return im_file


def add_info_to_save_path(save_path, info):
    """Add more information to save path"""
    fname, fextension = os.path.splitext(save_path)
    fname = '_'.join([fname, info])
    save_path = ''.join([fname, fextension])
    return save_path


def predict(model,
            model_path,
            image_list,
            transforms,
            thing_list,
            label_divisor,
            stuff_area,
            ignore_index,
            image_dir=None,
            save_dir='output',
            threshold=0.1,
            nms_kernel=7,
            top_k=200):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        image_list (list): A list of image path to be predicted.
        transforms (transform.Compose): Preprocess for input image.
        thing_list (list): A List of thing class id.
        label_divisor (int): An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        stuff_area (int): An Integer, remove stuff whose area is less tan stuff_area.
        ignore_index (int): Specifies a value that is ignored.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        threshold(float, optional): Threshold applied to center heatmap score. Defalut: 0.1.
        nms_kernel(int, optional): NMS max pooling kernel size. Default: 7.
        top_k(int, optional): Top k centers to keep. Default: 200.
    """
    paddleseg.utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    semantic_save_dir = os.path.join(save_dir, 'semantic')
    instance_save_dir = os.path.join(save_dir, 'instance')
    panoptic_save_dir = os.path.join(save_dir, 'panoptic')

    colormap = utils.cityscape_colormap()

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            ori_im = cv2.imread(im_path)
            ori_shape = ori_im.shape[:2]
            im, _ = transforms(ori_im)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            semantic, semantic_softmax, instance, panoptic, ctr_hmp = infer.inference(
                model=model,
                im=im,
                transforms=transforms.transforms,
                thing_list=thing_list,
                label_divisor=label_divisor,
                stuff_area=stuff_area,
                ignore_index=ignore_index,
                threshold=threshold,
                nms_kernel=nms_kernel,
                top_k=top_k,
                ori_shape=ori_shape)
            semantic = semantic.squeeze().numpy()
            instance = instance.squeeze().numpy()
            panoptic = panoptic.squeeze().numpy()

            im_file = get_save_name(im_path, image_dir)

            # visual semantic segmentation results
            save_path = os.path.join(semantic_save_dir, im_file)
            mkdir(save_path)
            utils.visualize_semantic(
                semantic, save_path=save_path, colormap=colormap)
            # Save added image for semantic segmentation results
            save_path_ = add_info_to_save_path(save_path, 'add')
            utils.visualize_semantic(
                semantic, save_path=save_path_, colormap=colormap, image=ori_im)
            # panoptic to semantic
            ins_mask = panoptic > label_divisor
            pan_to_sem = panoptic.copy()
            pan_to_sem[ins_mask] = pan_to_sem[ins_mask] // label_divisor
            save_path_ = add_info_to_save_path(save_path,
                                               'panoptic_to_semantic')
            utils.visualize_semantic(
                pan_to_sem, save_path=save_path_, colormap=colormap)
            save_path_ = add_info_to_save_path(save_path,
                                               'panoptic_to_semantic_added')
            utils.visualize_semantic(
                pan_to_sem,
                save_path=save_path_,
                colormap=colormap,
                image=ori_im)

            # vusual instance segmentation results
            pan_to_ins = panoptic.copy()
            ins_mask = pan_to_ins > label_divisor
            pan_to_ins[~ins_mask] = 0
            save_path = os.path.join(instance_save_dir, im_file)
            mkdir(save_path)
            utils.visualize_instance(pan_to_ins, save_path=save_path)
            # Save added image for instance segmentation results
            save_path_ = add_info_to_save_path(save_path, 'added')
            utils.visualize_instance(
                pan_to_ins, save_path=save_path_, image=ori_im)

            # visual panoptic segmentation results
            save_path = os.path.join(panoptic_save_dir, im_file)
            mkdir(save_path)
            utils.visualize_panoptic(
                panoptic,
                save_path=save_path,
                label_divisor=label_divisor,
                colormap=colormap,
                ignore_index=ignore_index)
            # Save added image for panoptic segmentation results
            save_path_ = add_info_to_save_path(save_path, 'added')
            utils.visualize_panoptic(
                panoptic,
                save_path=save_path_,
                label_divisor=label_divisor,
                colormap=colormap,
                image=ori_im,
                ignore_index=ignore_index)

            progbar_pred.update(i + 1)
