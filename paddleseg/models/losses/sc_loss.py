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

# import paddle
# import paddle.nn as nn
# import paddle.nn.functional as F

# from paddleseg.cvlibs import manager
# from paddleseg.models.losses.binary_cross_entropy_loss import BCELoss

# @manager.LOSSES.add_component
# class SCLoss(nn.Layer):
#     def __init__(sc_weight=0.2):
#         super().__init__()
#         if sc_weight>0:
#             self.semantic_centroid = paddle.Tensor()
#             self.semantic_centroid = paddle.create_parameter()
