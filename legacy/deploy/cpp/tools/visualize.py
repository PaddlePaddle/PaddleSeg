# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import sys

# ColorMap for visualization more clearly
color_map = [[128, 64, 128], [244, 35, 231], [69, 69, 69], [102, 102, 156],
             [190, 153, 153], [153, 153, 153], [250, 170, 29], [219, 219, 0],
             [106, 142, 35], [152, 250, 152], [69, 129, 180], [219, 19, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 69], [0, 60, 100], [0, 79, 100],
             [0, 0, 230], [119, 10, 32]]
# python visualize.py demo1.jpg demo1_jpg.png vis_result.png
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python visualize.py demo1.jpg demo1_jpg.png vis_result.png")
    else:
        ori_im = cv2.imread(sys.argv[1])
        ori_shape = ori_im.shape
        print(ori_shape)
        im = cv2.imread(sys.argv[2])
        shape = im.shape
        print("visualizing...")
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                im[i, j] = color_map[im[i, j, 0]]
        im = cv2.resize(im, (ori_shape[1], ori_shape[0]))
        cv2.imwrite(sys.argv[3], im)
        print("visualizing done!")
