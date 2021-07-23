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

import argparse
import codecs
import os

import yaml
import numpy as np
import paddleseg.transforms as T
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.utils.visualize import get_pseudo_color_map


import time
from paddleseg.utils import TimeAverager

class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(
            self.dic['Deploy']['transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)


class Predictor:
    def __init__(self, args):
        self.cfg = DeployConfig(args.cfg)
        self.args = args

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        if self.args.use_gpu:
            pred_cfg.enable_use_gpu(100, 0)

            if self.args.use_trt:
                ptype = PrecisionType.Int8 if args.use_int8 else PrecisionType.Float32
                pred_cfg.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=1,
                    min_subgraph_size=3,
                    precision_mode=ptype,
                    use_static=False,
                    use_calib_mode=False)

        self.predictor = create_predictor(pred_cfg)

        # 1.Init a timer
        self.cost_averager = TimeAverager()


    def preprocess(self, img):
        return self.cfg.transforms(img)[0]

    def run(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]

        num = len(imgs)
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        results = []

        logger.info('Got {} images, ready for inferencing'.format(num))
        logger.info('...')
        logger.info('..')
        logger.info('.')

        for i in range(0, num, self.args.batch_size):

            data = np.array([
                self.preprocess(img) for img in imgs[i:i + self.args.batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            # 2.Start
            start = time.time() 

            self.predictor.run()

            # 3.End
            self.cost_averager.record(time.time() - start)

            logger.info('The end of this inferencing. Time cost is : {} (s)'.format(time.time() - start))
            #print('The end of this inferencing. Time cost is : {} (s)'.format(time.time() - start))


            output_names = self.predictor.get_output_names()
            output_handle = self.predictor.get_output_handle(output_names[0])
            results.append(output_handle.copy_to_cpu())

        # 4.Return average time
        time_result = self.cost_averager.get_average()
        return time_result

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path of the image to be predicted.',
        type=str,
        default=None,
        required=True)
    
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)

    parser.add_argument(
        '--use_trt',
        dest='use_trt',
        help='Whether to use Nvidia TensorRT to accelerate prediction.',
        action='store_true')

    parser.add_argument(
        '--use_int8',
        dest='use_int8',
        help='Whether to use Int8 prediction when using TensorRT prediction.',
        action='store_true')

    parser.add_argument(
        '--image_num',
        dest='image_num',
        help='The number of images which used to caculate inference time.',
        default=100)

    return parser.parse_args()


def get_images(image_num,image_path, support_ext=".jpg|.jpeg|.png"):
    if not os.path.exists(image_path):
        raise Exception(f"Image path {image_path} invalid")

    # If given a file path.
    if os.path.isfile(image_path): 
        return [image_path]

    # If given a directory which contains images.
    imgs = []
    image_num = int(image_num)

    cnt = 0 # Record the readed image number.
    for item in os.listdir(image_path):
        if cnt < image_num: 
            cnt = cnt + 1
            ext = os.path.splitext(item)[1][1:].strip().lower()
            if (len(ext) > 0 and ext in support_ext):
                    item_path = os.path.join(image_path, item)
                    imgs.append(item_path)
        else:
            break # The rest part is no need to read.

    return imgs


def main(args):
    # Show the configs
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    # Create predictor for exported model
    args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else False
    predictor = Predictor(args)

    # Do inference 
    time_result = predictor.run(get_images(args.image_num,args.image_path))
    FPS = 1 / time_result
    
    logger.info('Perform a total of {} inferences，the average inference time is : {} (s), FPS is : {}'.format(args.image_num,time_result,FPS))


if __name__ == '__main__':
    args = parse_args()
    main(args)
