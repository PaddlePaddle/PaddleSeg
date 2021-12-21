# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
This code is based on https://github.com/saic-vul/ritm_interactive_segmentation
Ths copyright of saic-vul/ritm_interactive_segmentation is as follows:
MIT License [see LICENSE for details]
"""

from .base import BasePredictor
from inference.transforms import ZoomIn


def get_predictor(
    net, brs_mode, with_flip=False, zoom_in_params=dict(), predictor_params=None
):

    predictor_params_ = {"optimize_after_n_clicks": 1}

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if brs_mode == "NoBRS":

        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(
            net, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_
        )

    else:
        raise NotImplementedError("Just support NoBRS mode")
    return predictor
