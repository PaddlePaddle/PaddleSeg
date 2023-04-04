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

# TODO: Make the encoding system object-oriented and configurable.

_CAT_ID_OFFSET = 1


def decode_pan_id(pan_id, label_divisor):
    return (pan_id // label_divisor) - _CAT_ID_OFFSET, pan_id % label_divisor


def encode_pan_id(cat_id, label_divisor, ins_id=0):
    return (cat_id + _CAT_ID_OFFSET) * label_divisor + ins_id


def is_crowd(pan_id, label_divisor):
    return pan_id % label_divisor == 0
