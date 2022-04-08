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

from .zoom_in import ZoomIn, get_roi_image_nd


class LimitLongestSide(ZoomIn):
    def __init__(self, max_size=800):
        super().__init__(target_size=max_size, skip_clicks=0)

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        image_max_size = max(image_nd.shape[2:4])
        self.image_changed = False

        if image_max_size <= self.target_size:
            return image_nd, clicks_lists
        self._input_image = image_nd

        self._object_roi = (0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1)
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi,
                                           self.target_size)
        self.image_changed = True

        tclicks_lists = [self._transform_clicks(clicks_lists[0])]
        return self._roi_image, tclicks_lists
