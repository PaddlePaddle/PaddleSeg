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

from .mixed_loss import MixedLoss
from .cross_entropy_loss import CrossEntropyLoss
from .cross_entropy_loss import DistillCrossEntropyLoss
from .binary_cross_entropy_loss import BCELoss
from .lovasz_loss import LovaszSoftmaxLoss, LovaszHingeLoss
from .gscnn_dual_task_loss import DualTaskLoss
from .edge_attention_loss import EdgeAttentionLoss
from .bootstrapped_cross_entropy import BootstrappedCrossEntropyLoss
from .dice_loss import DiceLoss
from .ohem_cross_entropy_loss import OhemCrossEntropyLoss
from .decoupledsegnet_relax_boundary_loss import RelaxBoundaryLoss
from .ohem_edge_attention_loss import OhemEdgeAttentionLoss
from .l1_loss import L1Loss
from .mean_square_error_loss import MSELoss
from .focal_loss import FocalLoss
from .kl_loss import KLLoss
from .rmi_loss import RMILoss
from .detail_aggregate_loss import DetailAggregateLoss
from .point_cross_entropy_loss import PointCrossEntropyLoss
from .pixel_contrast_cross_entropy_loss import PixelContrastCrossEntropyLoss
from .semantic_encode_cross_entropy_loss import SECrossEntropyLoss
from .semantic_connectivity_loss import SemanticConnectivityLoss
