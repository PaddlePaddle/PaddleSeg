# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class _ArgParserWrapper(object):
    def __init__(self, parser):
        super().__init__()
        self._parser = parser
        self._compat_parser = self._build_compat_parser()

    def parse_args(self, *args, **kwargs):
        args, remainder = self._parser.parse_known_args(*args, **kwargs)
        compat_args = self._compat_parser(remainder)
        # `args` takes higher precedence
        compat_args.update(args)
        return compat_args

    def _build_compat_parser(self):
        raise NotImplementedError


class TrainArgParserWrapper(_ArgParserWrapper):
    def _build_compat_parser(self):
        parser = argparse.ArgumentParser()

        # Common params
        parser.add_argument(
            '--device',
            help='Set the device place for training model.',
            default='gpu',
            choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
            type=str)
        parser.add_argument(
            '--save_dir',
            help='The directory for saving the model snapshot.',
            type=str,
            default='./output')
        parser.add_argument(
            '--num_workers',
            help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
            type=int,
            default=0)
        parser.add_argument(
            '--do_eval',
            help='Whether to do evaluation in training.',
            action='store_true')
        parser.add_argument(
            '--use_vdl',
            help='Whether to record the data to VisualDL in training.',
            action='store_true')

        # Runntime params
        parser.add_argument(
            '--resume_model',
            help='The path of the model to resume training.',
            type=str)
        parser.add_argument('--iters', help='Iterations in training.', type=int)
        parser.add_argument(
            '--batch_size',
            help='Mini batch size of one gpu or cpu. ',
            type=int)
        parser.add_argument(
            '--learning_rate', help='Learning rate.', type=float)
        parser.add_argument(
            '--save_interval',
            help='How many iters to save a model snapshot once during training.',
            type=int,
            default=1000)
        parser.add_argument(
            '--log_iters',
            help='Display logging information at every `log_iters`.',
            default=10,
            type=int)
        parser.add_argument(
            '--keep_checkpoint_max',
            help='Maximum number of checkpoints to save.',
            type=int,
            default=5)

        # Other params
        parser.add_argument(
            '--seed',
            help='Set the random seed in training.',
            default=None,
            type=int)
        parser.add_argument(
            "--precision",
            default="fp32",
            type=str,
            choices=["fp32", "fp16"],
            help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
        )
        parser.add_argument(
            "--amp_level",
            default="O1",
            type=str,
            choices=["O1", "O2"],
            help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                    data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                    parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                    and batchnorm. Default is O1(amp).")
        parser.add_argument(
            '--profiler_options',
            type=str,
            help='The option of train profiler. If profiler_options is not None, the train ' \
                'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
        )
        parser.add_argument(
            '--data_format',
            help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
            type=str,
            default='NCHW')
        parser.add_argument(
            '--repeats',
            type=int,
            default=1,
            help="Repeat the samples in the dataset for `repeats` times in each epoch."
        )

        return parser
