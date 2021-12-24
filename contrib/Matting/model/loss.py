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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
import cv2


@manager.LOSSES.add_component
class MRSD(nn.Layer):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64.
            label (Tensor): Label tensor, the data type is float32, float64. The shape should equal to logit.
            mask (Tensor, optional): The mask where the loss valid. Default： None.
        """
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        sd = paddle.square(logit - label)
        loss = paddle.sqrt(sd + self.eps)
        if mask is not None:
            mask = mask.astype('float32')
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + self.eps)
            mask.stop_gradient = True
        else:
            loss = loss.mean()

        return loss

@manager.LOSSES.add_component
class GradientLoss(nn.Layer):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.kernel_x, self.kernel_y = self.sobel_kernel()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            logit = logit * mask
            label = label * mask
            loss = paddle.sum(F.l1_loss(self.sobel(logit), self.sobel(label), 'none')) / (mask.sum() + self.eps)
        else:
            loss = F.l1_loss(self.sobel(logit), self.sobel(label), 'mean')

        return loss

    def sobel(self, input):
        """Using Sobel to compute gradient. Return the magnitude."""
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect NCHW, but it is ", input.shape)
        
        n, c, h, w = input.shape

        input_pad = paddle.reshape(input, (n*c, 1, h, w))
        input_pad = F.pad(input_pad, pad=[1, 1, 1, 1], mode='replicate')

        grad_x = F.conv2d(input_pad, self.kernel_x, padding=0)
        grad_y = F.conv2d(input_pad, self.kernel_y, padding=0)

        mag = paddle.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        mag = paddle.reshape(mag, (n, c, h, w))

        return mag

    def sobel_kernel(self):
        kernel_x = paddle.to_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).astype('float32')
        kernel_x = kernel_x / kernel_x.abs().sum()
        kernel_y = kernel_x.transpose([1, 0])
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        kernel_x.stop_gradient = True
        kernel_y.stop_gradient = True
        return kernel_x, kernel_y


if __name__ == "__main__":
    gradiend_loss = GradientLoss()
    import cv2
    import numpy as np
    alpha = cv2.imread('/home/aistudio/data/Distinctions-646/val/alpha/test_0.png', 0)
    alpha = alpha / 255.
    h, w = alpha.shape
    pred = np.clip(np.random.rand(h ,w)/5. + alpha, 0., 1.)
    alpha = paddle.to_tensor(alpha)
    pred = paddle.to_tensor(pred)
    alpha = alpha.unsqueeze(0).unsqueeze(0).astype('float32')
    pred = pred.unsqueeze(0).unsqueeze(0).astype('float32')
    print(alpha.shape, pred.shape)
    loss = gradiend_loss(pred, alpha)
    print(loss)