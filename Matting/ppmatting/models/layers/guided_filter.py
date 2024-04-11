# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BoxFilter(nn.Layer):

    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, x):
        """
        Using convolution for box blur like RVM: <https://github.com/PeterL1n/RobustVideoMatting/>
        The original implementation: <https://github.com/wuhuikai/DeepGuidedFilter/>
        """
        kernel_size = 2 * self.r + 1
        kernel_x = paddle.full((x.shape[1], 1, 1, kernel_size), 1 / kernel_size)
        kernel_y = paddle.full((x.shape[1], 1, kernel_size, 1), 1 / kernel_size)
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.shape[1])

        return x


class FastGuidedFilter(nn.Layer):

    def __init__(self, r, eps=1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfileter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        """
        lr_x and lr_y should be the same shape, except lr_x[1] == 1.
        The heigh and width of hr_x should be larger than lr_x and have the same channels.
        """
        mean_x = self.boxfileter(lr_x)
        mean_y = self.boxfileter(lr_y)
        cov_xy = self.boxfileter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfileter(lr_x * lr_x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        A = F.interpolate(A,
                          hr_x.shape[2:],
                          mode='bilinear',
                          align_corners=False)
        b = F.interpolate(b,
                          hr_x.shape[2:],
                          mode='bilinear',
                          align_corners=False)

        return A * hr_x + b
