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
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        smooth (float32): laplace smoothing,
            to smooth dice loss and accelerate convergence. following:
            https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
    """

    def __init__(self, ignore_index=255, smooth=0.):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5
        self.smooth = smooth

    def forward(self, logits, labels):
        labels = paddle.cast(labels, dtype='int32')
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1])
        labels_one_hot = paddle.transpose(labels_one_hot, [0, 3, 1, 2])
        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')

        logits = F.softmax(logits, axis=1)

        mask = (paddle.unsqueeze(labels, 1) != self.ignore_index)
        logits = logits * mask
        labels_one_hot = labels_one_hot * mask

        dims = (0, ) + tuple(range(2, labels.ndimension() + 1))

        intersection = paddle.sum(logits * labels_one_hot, dims)
        cardinality = paddle.sum(logits + labels_one_hot, dims)
        dice_loss = ((2. * intersection + self.smooth) /
                     (cardinality + self.eps + self.smooth)).mean()
        return 1 - dice_loss
