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


@manager.LOSSES.add_component
class DualTaskLoss(nn.Layer):
    """
    The dual task loss implement of GSCNN

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        tau (float): the tau of gumbel softmax sample.
    """

    def __init__(self, ignore_index=255, tau=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.tau = tau

    def _gumbel_softmax_sample(self, logit, tau=1, eps=1e-10):
        """
        Draw a sample from the Gumbel-Softmax distribution

        based on
        https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
        (MIT license)
        """
        gumbel_noise = paddle.rand(logit.shape)
        gumbel_noise = -paddle.log(eps - paddle.log(gumbel_noise + eps))
        logit = logit + gumbel_noise
        return F.softmax(logit / tau, axis=1)

    def compute_grad_mag(self, x):
        eps = 1e-6
        n, c, h, w = x.shape
        if h <= 1 or w <= 1:
            raise ValueError(
                'The width and height of tensor to compute grad must be greater than 1, but the shape is {}.'
                .format(x.shape))

        x = self.conv_tri(x, r=4)
        kernel = [[-1, 0, 1]]
        kernel = paddle.to_tensor(kernel).astype('float32')
        kernel = 0.5 * kernel

        kernel_x = paddle.concat([kernel.unsqueeze((0, 1))] * c, axis=0)
        grad_x = F.conv2d(x, kernel_x, padding='same', groups=c)
        kernel_y = paddle.concat([kernel.t().unsqueeze((0, 1))] * c, axis=0)
        grad_y = F.conv2d(x, kernel_y, padding='same', groups=c)
        mag = paddle.sqrt(grad_x * grad_x + grad_y * grad_y + eps)

        return mag / mag.max()

    def conv_tri(self, input, r):
        """
        Convolves an image by a 2D triangle filter (the 1D triangle filter f is
        [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
        """
        if r <= 1:
            raise ValueError(
                '`r` should be greater than 1, but it is {}.'.format(r))

        kernel = [
            list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
        ]
        kernel = paddle.to_tensor(kernel).astype('float32')
        kernel = kernel / (r + 1)**2
        input_ = F.pad(input, [1, 1, 0, 0], mode='replicate')
        input_ = F.pad(input_, [r, r, 0, 0], mode='reflect')
        input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
        input_ = paddle.concat(input_, axis=3)
        tem = input_.clone()

        input_ = F.pad(input_, [0, 0, 1, 1], mode='replicate')
        input_ = F.pad(input_, [0, 0, r, r], mode='reflect')
        input_ = [input_[:, :, :r, :], tem, input_[:, :, -r:, :]]
        input_ = paddle.concat(input_, axis=2)

        c = input.shape[1]
        kernel_x = paddle.concat([kernel.unsqueeze((0, 1))] * c, axis=0)
        output = F.conv2d(input_, kernel_x, padding=0, groups=c)
        kernel_y = paddle.concat([kernel.t().unsqueeze((0, 1))] * c, axis=0)
        output = F.conv2d(output, kernel_y, padding=0, groups=c)
        return output

    def forward(self, logit, labels):
        # import pdb; pdb.set_trace()
        n, c, h, w = logit.shape
        th = 1e-8
        eps = 1e-10
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1)
        mask = (labels != self.ignore_index)
        mask.stop_gradient = True
        logit = logit * mask

        labels = labels * mask
        if len(labels.shape) == 4:
            labels = labels.squeeze(1)
        labels.stop_gradient = True
        labels = F.one_hot(labels, logit.shape[1]).transpose((0, 3, 1, 2))
        labels.stop_gradient = True

        g = self._gumbel_softmax_sample(logit, tau=self.tau)
        g = self.compute_grad_mag(g)
        g_hat = self.compute_grad_mag(labels)
        loss = F.l1_loss(g, g_hat, reduction='none')
        loss = loss * mask

        g_mask = (g > th).astype('float32')
        g_mask.stop_gradient = True
        g_mask_sum = paddle.sum(g_mask)
        loss_g = paddle.sum(loss * g_mask)
        if g_mask_sum > eps:
            loss_g = loss_g / g_mask_sum

        g_hat_mask = (g_hat > th).astype('float32')
        g_hat_mask.stop_gradient = True
        g_hat_mask_sum = paddle.sum(g_hat_mask)
        loss_g_hat = paddle.sum(loss * g_hat_mask)
        if g_hat_mask_sum > eps:
            loss_g_hat = loss_g_hat / g_hat_mask_sum

        total_loss = 0.5 * loss_g + 0.5 * loss_g_hat

        return total_loss
