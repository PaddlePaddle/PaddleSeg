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
import json
import random
import tempfile
import filelock
import contextlib
import numpy as np
import SimpleITK as sitk
from collections.abc import Iterable
from urllib.parse import urlparse, unquote
from functools import partial, update_wrapper

import paddle
from medicalseg.utils import logger, seg_env
from medicalseg.utils.download import download_file_and_uncompress


@contextlib.contextmanager
def generate_tempdir(directory: str=None, **kwargs):
    '''Generate a temporary directory'''
    directory = seg_env.TMP_HOME if not directory else directory
    with tempfile.TemporaryDirectory(dir=directory, **kwargs) as _dir:
        yield _dir


def load_entire_model(model, pretrained):
    if pretrained is not None:
        load_pretrained_model(model, pretrained)
    else:
        logger.warning('Not all pretrained params of {} are loaded, ' \
                       'training from scratch or a pretrained backbone.'.format(model.__class__.__name__))


def download_pretrained_model(pretrained_model):
    """
    Download pretrained model from url.
    Args:
        pretrained_model (str): the url of pretrained weight
    Returns:
        str: the path of pretrained weight
    """
    assert urlparse(pretrained_model).netloc, "The url is not valid."

    pretrained_model = unquote(pretrained_model)
    savename = pretrained_model.split('/')[-1]
    if not savename.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        savename = pretrained_model.split('/')[-2]
    else:
        savename = savename.split('.')[0]

    with generate_tempdir() as _dir:
        with filelock.FileLock(os.path.join(seg_env.TMP_HOME, savename)):
            pretrained_model = download_file_and_uncompress(
                pretrained_model,
                savepath=_dir,
                extrapath=seg_env.PRETRAINED_MODEL_HOME,
                extraname=savename)
            pretrained_model = os.path.join(pretrained_model, 'model.pdparams')
    return pretrained_model


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
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
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


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            iter = resume_model.split('_')[-1]
            iter = int(iter)
            return iter
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 100000))


def get_image_list(image_path, valid_suffix=None, filter_key=None):
    """Get image list from image name or image directory name with valid suffix.

    if needed, filter_key can be used to whether 'include' the key word.
    When filter_key is not None，it indicates whether filenames should include certain key.


    Args:
    image_path(str): the image or image folder where you want to get a image list from.
    valid_suffix(tuple): Contain only the suffix you want to include.
    filter_key(dict): the key(ignore case) and whether you want to include it. e.g.:{"segmentation": True} will futher filter the imagename with segmentation in it.

    """
    if valid_suffix is None:
        valid_suffix = [
            'nii.gz', 'nii', 'dcm', 'nrrd', 'mhd', 'raw', 'npy', 'mha'
        ]

    image_list = []
    if os.path.isfile(image_path):
        if image_path.split("/")[-1].split('.', maxsplit=1)[-1] in valid_suffix:
            if filter_key is not None:
                f_name = image_path.split("/")[
                    -1]  # TODO change to system invariant
                for key, val in filter_key.items():
                    if (key in f_name.lower()) is not val:
                        break
                else:
                    image_list.append(image_path)
            else:
                image_list.append(image_path)
        else:
            raise FileNotFoundError(
                '{} is not a file end with supported suffix, the support suffixes are {}.'
                .format(image_path, valid_suffix))

    # load image in a directory
    elif os.path.isdir(image_path):
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if f.split(".", maxsplit=1)[-1] in valid_suffix:
                    if filter_key is not None:
                        for key, val in filter_key.items():
                            if (key in f.lower()) is not val:
                                break
                        else:
                            image_list.append(os.path.join(root, f))
                    else:
                        image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '{} is not found. it should be a path of image, or a directory including images.'.
            format(image_path))

    if len(image_list) == 0:
        raise RuntimeError(
            'There are not image file in `--image_path`={}'.format(image_path))

    return image_list


def save_array(save_path, save_content, form, image_infor):
    """
    save_path: Example: save_dir/iter,
    save_content: dict of saveing content, where key is the name and value is the content. 
                 Example: {'pred': pred.numpy(), 'label': label.numpy(), 'img': im.numpy()}
    form: Iterable that containing the format want to save.('npy', 'nii.gz')
    image_infor: Dict containing the information needed to save the image.
                Example: {spacing: xx, direction: xx, origin: xx, format: 'zyx'}
    """
    if not isinstance(save_content, dict):
        raise TypeError(
            'The save_content need to be dict which the key is the save name and the value is the numpy array to be saved, but recieved {}'
            .format(type(save_content)))

    for (key, val) in save_content.items():
        if not isinstance(val, np.ndarray):
            raise TypeError('We only save numpy array, but recieved {}'.format(
                type(val)))
        if len(val.shape) > 3:
            save_content[key] = np.squeeze(val)

    if not isinstance(form, Iterable):
        raise TypeError('The form need be iterable, but recieved {}'.format(
            type(form)))

    if save_path is not None:
        for suffix in form:
            if suffix == 'npy':
                for (key, val) in save_content.items():
                    np.save('{}_{}.npy'.format(save_path, key), val)
            elif suffix == 'nii' or suffix == 'nii.gz':
                for (key, val) in save_content.items():
                    if image_infor["format"] == "xyz":
                        val = np.transpose(val, [2, 1, 0])
                    elif image_infor["format"] != "zyx":
                        raise RuntimeError(
                            "the image format {} is not supported".format(
                                image_infor["format"]))

                    img_itk_new = sitk.GetImageFromArray(val)
                    img_itk_new.SetSpacing(tuple(image_infor["spacing"]))
                    img_itk_new.SetOrigin(tuple(image_infor["origin"]))
                    img_itk_new.SetDirection(tuple(image_infor["direction"]))
                    sitk.WriteImage(
                        img_itk_new,
                        os.path.join('{}_{}.{}'.format(save_path, key, suffix)))
            else:
                raise RuntimeError(
                    'Save format other than npy or nii/nii.gz is not supported yet.'
                )

        print("[EVAL] Sucessfully save to {}".format(save_path))


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
