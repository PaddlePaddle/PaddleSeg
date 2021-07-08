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
import numpy as np


@manager.LOSSES.add_component
class FocalLoss(nn.Layer):
    """
    The Focal Loss implement of Portrait Net.

    Args:
        gamma (float): the coefficient of Focal Loss.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
    def __init__(self, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inp, target):
        inp = paddle.reshape(inp, [inp.shape[0], inp.shape[1], -1])
        inp = paddle.transpose(inp, [0, 2, 1])
        inp = paddle.reshape(inp, [-1, 2])
        target = paddle.reshape(target, [-1, 1])
        range_ = paddle.arange(0, target.shape[0])
        range_ = paddle.unsqueeze(range_, axis=-1)
        target = paddle.cast(target, dtype='int64')
        target = paddle.concat([range_, target], axis=-1)
        logpt = F.log_softmax(inp)
        logpt = paddle.gather_nd(logpt, target)

        pt = paddle.to_tensor(np.exp(logpt.numpy()))
        loss = -1 * (1 - pt)**self.gamma * logpt
        loss = paddle.mean(loss)
        return loss
