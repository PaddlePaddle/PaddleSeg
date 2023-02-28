# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from abc import abstractmethod, ABCMeta

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class _MatrixDecomposition2DBase(nn.Layer, metaclass=ABCMeta):
    """
    The base implementation of 2d matrix decomposition.

    The original article refers to
    Yuanduo Hong, Huihui Pan, Weichao Sun, et al. "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
    (https://arxiv.org/abs/2101.06085)
    """

    def __init__(self, args=None):
        super().__init__()
        if args is None:
            args = dict()
        elif not isinstance(args, dict):
            raise TypeError("`args` must be a dict, but got {}".foramt(
                args.__class__.__name__))

        self.spatial = args.setdefault("SPATIAL", True)

        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)

        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)

        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)

        self.rand_init = args.setdefault("RAND_INIT", True)

    @abstractmethod
    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    @abstractmethod
    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @abstractmethod
    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = paddle.bmm(x.transpose([0, 2, 1]), bases)
        coef = F.softmax(self.inv_t * coef, axis=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.reshape([B * self.S, D, N])
        else:
            D = H * W
            N = C // self.S
            x = x.reshape([B * self.S, N, D]).transpose([0, 2, 1])

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R)
        else:
            bases = paddle.repeat_interleave(self.bases, B, 0)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = paddle.bmm(bases, coef.transpose([0, 2, 1]))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.reshape([B, C, H, W])
        else:
            x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        # (B * H, D, R) -> (B, H, N, D)
        # bases = bases.reshape([B, self.S, D, self.R])

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = paddle.rand((B * S, D, R))

        bases = F.normalize(bases, axis=1)

        return bases

    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = paddle.bmm(x.transpose([0, 2, 1]), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose([0, 2, 1]).bmm(bases))

        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = paddle.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose([0, 2, 1]).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = paddle.bmm(x.transpose([0, 2, 1]), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose([0, 2, 1]).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef
