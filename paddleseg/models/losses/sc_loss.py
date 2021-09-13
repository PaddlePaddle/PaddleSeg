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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models.losses.binary_cross_entropy_loss import BCELoss


@manager.LOSSES.add_component
class SCLoss(nn.Layer):
    def __init__(self, ignore_index=255, nclass=60):
        super(SCLoss, self).__init__()
        self.nclass = nclass
        self.EPS = 1e-8
        self.weight = None
        self.ignore_index = ignore_index
        self.semantic_centroid = paddle.ones(shape=(nclass, 256))
        self.semantic_centroid = paddle.create_parameter(
                        shape=self.semantic_centroid.shape,
                        dtype=str(self.semantic_centroid.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(self.semantic_centroid))
    
    def forward(self, logit, label):
        sc_out = logit
        sc_out = (sc_out.transpose((0, 2, 1)) * self.semantic_centroid).sum(axis=2)
        se_target = self._get_batch_label_vector(target=label, nclass=self.nclass).astype('float32')

        loss = F.binary_cross_entropy(F.sigmoid(sc_out), se_target)

        return self._post_process_loss(logit, label, loss)

    def _post_process_loss(self, logit, label, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        label.stop_gradient = True
        mask.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask

        if self.weight is not None:
            _one_hot = F.one_hot(label, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)


        return avg_loss


    @staticmethod
    def _get_batch_label_vector(target, nclass):
        """
        transfer target of shape B*H*W to B*nclass
        """
        batch = target.shape[0]
        tvect = paddle.zeros([batch, nclass])
        for i in range(batch):
            hist = paddle.histogram(target[i], bins=nclass, min=0, max=nclass-1)
            # hist = paddle.histogram(target[i].cpu().data.float(), bins=nclass, min=0, max=nclass-1)

            vect = hist > 0
            tvect[i] = vect.astype('float32')
        
        return tvect


if __name__ == '__main__':
    scloss = SCLoss()
    logit = paddle.rand(shape=(60, 256, 256))
    Label = paddle.randint(low=0,high=20, shape=(1,256,256))
    loss = scloss(logit, Label)
    print('loss is %.5f'% loss)

            
 