# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os

import numpy as np
import shutil
from paddleseg.cvlibs import manager, Config
import paddle
from paddleslim.quant import quant_post_static

paddle.enable_static()


def parse_args():
    parser = argparse.ArgumentParser(description='Model quant')

    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The directory of model',
        type=str,
        default=None)
    parser.add_argument(
        '--batch_nums',
        dest='batch_nums',
        help='number of iterations. If set to None , it will run until the end of the sample_generator iteration, otherwise, the number of iterations is batch_nums, that is, the number of samples involved in Scale correction is batch_nums * batch_size .',
        type=int,
        default=10)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Number of pictures per batch',
        type=int,
        default=1)

    return parser.parse_args()


def sample_generator(loader):
    def __reader__():
        for indx, data in enumerate(loader):
            images = np.array(data['img'])
            yield images

    return __reader__


def main(args):
    fp32_model_dir = args.model_dir
    quant_output_dir = 'quant_model'
    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    use_gpu = True
    place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    data_loader = paddle.io.DataLoader(
        val_dataset, places=place, drop_last=False, batch_size=1, shuffle=False)

    quant_post_static(
        executor=exe,
        model_dir=fp32_model_dir,
        quantize_model_path=quant_output_dir,
        save_model_filename='model.pdmodel',
        save_params_filename='model.pdiparams',
        sample_generator=sample_generator(data_loader),
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        batch_size=args.batch_size,
        batch_nums=args.batch_nums,
        algo='KL')

    # copy yml
    shutil.copyfile(
        os.path.join(fp32_model_dir, 'deploy.yaml'),
        os.path.join(quant_output_dir, 'deploy.yaml'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
