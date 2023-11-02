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

# This implementation refers to: https://github.com/facebookresearch/segment-anything

import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry

model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams",
    'vit_t':
    "https://paddleseg.bj.bcebos.com/dygraph/paddlesegAnything/vit_t/model.pdparam"
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Segment image with point promp or box')
    # Parameters
    parser.add_argument(
        '--input_path', type=str, required=True, help='The directory of image.')
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_l",
        required=True,
        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b', 'vit_t']",
    )
    parser.add_argument(
        '--point_prompt',
        type=int,
        nargs='+',
        default=None,
        help='point prompt.')
    parser.add_argument(
        '--box_prompt',
        type=int,
        nargs='+',
        default=None,
        help='box prompt format as xyxy.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/',
        help='The directory for saving the results')
    return parser.parse_args()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def main(args):
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    input_path = args.input_path
    output_path = args.output_path
    point, box = args.point_prompt, args.box_prompt
    if point is not None:
        point = np.array([point])
        input_label = np.array([1])
    else:
        input_label = None
    if box is not None:
        box = np.array([[box[0], box[1]], [box[2], box[3]]])

    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = sam_model_registry[args.model_type](
        checkpoint=model_link[args.model_type])
    predictor = SamPredictor(model)
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=point,
        point_labels=input_label,
        box=box,
        multimask_output=True, )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    plt.axis('off')
    basename = os.path.basename(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    path_output = os.path.join(output_path, basename)
    plt.savefig(path_output)
    print('The output has been saved to {}'.format(path_output))


if __name__ == "__main__":
    args = get_args()
    main(args)
