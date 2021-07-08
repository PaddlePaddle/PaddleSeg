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

import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class KLLoss(nn.Layer):
    """
    The Kullback-Leibler divergence Loss implement of Portrait Net.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        temperature (float): the coefficient of kl_loss.
    """
    def __init__(self, ignore_index=255, temperature=1):
        super(KLLoss, self).__init__()
        self.ignore_index = ignore_index
        self.kl_loss = nn.KLDivLoss(reduction="mean")
        self.temperature = temperature

    def forward(self, inp, target):
        inp = F.log_softmax(inp / self.temperature, axis=1)
        target = F.softmax(target / self.temperature, axis=1)
        return self.kl_loss(inp, target) * self.temperature * self.temperature
