# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import, division, print_function

import os
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import paddle

from qinspector.apis.pipeline import Pipeline
from qinspector.cvlib.configs import ArgsParser


def argsparser():
    parser = ArgsParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path of configure"),
        required=True)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Suport path of one image file and the directory of images.",
        required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--device",
        type=str,
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        help="Choose the device you want to run, it can be: cpu/gpu/xpu/npu/mlu, default is gpu."
    )
    return parser.parse_args()


def main():
    args = argsparser()
    if args.device == 'gpu' and paddle.is_compiled_with_cuda():
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    elif args.device == 'mlu' and paddle.is_compiled_with_mlu():
        place = 'mlu'
    else:
        place = 'cpu'
    paddle.set_device(place)

    inputs = os.path.abspath(args.input)
    pipeline = Pipeline(args)
    result = pipeline.run(inputs)
    return result


if __name__ == '__main__':
    main()
