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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DiceLoss(nn.Layer):
    """
    The implements of the dice loss.

    Args:
        weight (list[float], optional): The weight for each class. Default: None.
        ignore_index (int64): ignore_index (int64, optional): Specifies a target value that
            is ignored and does not contribute to the input gradient. Default ``255``.
        smooth (float32): Laplace smoothing to smooth dice loss and accelerate convergence.
            Default: 1.0
    """

    def __init__(self, weight=None, ignore_index=255, smooth=1.0):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = 1e-8

    def forward(self, logits, labels):
        num_class = logits.shape[1]
        if self.weight is not None:
            assert num_class == len(self.weight), \
                "The lenght of weight should be euqal to the num class"

        mask = labels != self.ignore_index
        mask = paddle.cast(paddle.unsqueeze(mask, 1), 'float32')

        labels[labels == self.ignore_index] = 0
        labels_one_hot = F.one_hot(labels, num_class)
        labels_one_hot = paddle.transpose(labels_one_hot, [0, 3, 1, 2])
        logits = F.softmax(logits, axis=1)

        dice_loss = 0.0
        for i in range(num_class):
            dice_loss_i = dice_loss_helper(logits[:, i], labels_one_hot[:, i],
                                           mask, self.smooth, self.eps)
            if self.weight is not None:
                dice_loss_i *= self.weight[i]
            dice_loss += dice_loss_i
        dice_loss = dice_loss / num_class

        return dice_loss


def dice_loss_helper(logit, label, mask, smooth, eps):
    assert logit.shape == label.shape, \
        "The shape of logit and label should be the same"
    logit = paddle.reshape(logit, [0, -1])
    label = paddle.reshape(label, [0, -1])
    mask = paddle.reshape(mask, [0, -1])
    logit *= mask
    label *= mask
    intersection = paddle.sum(logit * label, axis=1)
    cardinality = paddle.sum(logit + label, axis=1)
    dice_loss = 1 - (2 * intersection + smooth) / (cardinality + smooth + eps)
    dice_loss = dice_loss.mean()
    return dice_loss
