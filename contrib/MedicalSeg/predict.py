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
from tqdm import tqdm
import argparse
import os
# from tools.prepare_acdc import *
import paddle
from medicalseg.core.infer import sliding_window_inference
from medicalseg.cvlibs import Config
from medicalseg.core import evaluate
from medicalseg.utils import get_sys_env, logger, config_check, utils
import nibabel as nib
import numpy as np
from skimage.transform import resize

def resize_image(image,new_shape,order=3,cval=0):
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    return resize(image,new_shape,order,cval=cval,**kwargs)

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default="saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model/model.pdparams"
    )
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')


    parser.add_argument('--sw_num', default=None, type=int, help='sw_num')


    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    # losses = cfg.loss
    # test_dataset = cfg.test_dataset

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')
    model.eval()
    image_path=args.image_path
    save_path=args.save_dir
    os.makedirs(save_path,exist_ok=True)
    imagefiles=[file for file in os.listdir(image_path) if file.endswith(".nii.gz")]
    new_spacing = [1.52, 1.52, 6.35]
    with paddle.no_grad():
        for filename in tqdm(imagefiles):
            nimg = nib.load(os.path.join(image_path, filename))
            data_array = nimg.get_data()
            # import pdb
            # pdb.set_trace()
            original_spacing = nimg.header["pixdim"][1:4]
            shape=data_array.shape
            new_shape = np.round(((np.array(original_spacing) / np.array(new_spacing)).astype(float) * np.array(shape))).astype(
                int)
            data_array = resize_image(data_array, new_shape)
            # 将数据从hwd转化为dhw
            # import pdb
            # pdb.set_trace()
            data_array=np.transpose(data_array,[2,0,1])

            mean=np.mean(data_array)
            std=np.std(data_array)

            if std>0:
                data_array=(data_array-mean)/std
            else:
                data_array=(data_array-mean)/(std+1e-8)
            data_array=paddle.to_tensor(data_array.astype("float32")).unsqueeze(0).unsqueeze(0)
            # import pdb
            # pdb.set_trace()
            logits=sliding_window_inference(data_array, model.img_shape, 1, model)
            # import pdb
            # pdb.set_trace()
            logit=logits[0]
            if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
                logit = logit.transpose((0, 4, 1, 2, 3))
            label_array = paddle.argmax(logit, axis=1, dtype='int32')

            label_array=np.transpose(label_array[0].numpy(),[1,2,0])
            label_array = resize_segmentation(label_array, shape)
            nlabel=nib.Nifti1Image(label_array, nimg.affine, header=nimg.header)

            nib.save(nlabel,os.path.join(save_path,filename))

if __name__ == '__main__':
    args = parse_args()
    main(args)