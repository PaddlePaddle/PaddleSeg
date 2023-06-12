# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
paddle_version = paddle.__version__[:3]
# paddle version < 2.5.0 and not develop
if paddle_version not in ["2.5", "0.0"]:
    from paddle.fluid.dataloader.collate import default_collate_fn
# paddle version >= 2.5.0 or develop
else:
    from paddle.io.dataloader.collate import default_collate_fn


class SegCollate(object):
    def __init__(self, batch_aug_fn=None):
        self.batch_aug_fn = batch_aug_fn

    def __call__(self, batch):
        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)
        return default_collate_fn(batch)
