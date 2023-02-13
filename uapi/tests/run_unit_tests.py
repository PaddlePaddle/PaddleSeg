#!/usr/bin/env python

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

import os
import unittest

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    ut_dir = os.path.join(__dir__, 'ut')

    loader = unittest.TestLoader()
    suite = loader.discover(ut_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)
