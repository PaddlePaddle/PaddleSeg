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
import sys

from uapi.base.utils.misc import run_cmd

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    smoke_test_dir = os.path.join(__dir__, 'smoke')

    for filename in os.listdir(smoke_test_dir):
        if filename.endswith('.py'):
            # Skip .py files
            continue
        full_path = os.path.join(smoke_test_dir, filename)
        if os.path.isdir(full_path):
            # Find scripts in directories
            model_type = os.path.basename(full_path)
            print("#" * 30)
            print(f"Run {model_type} smoke tests.")
            print("#" * 30)
            print("")
            for script_filename in os.listdir(full_path):
                script_path = os.path.join(full_path, script_filename)
                assert script_path.endswith('.py')
                cmd = f"{sys.executable} {script_path}"
                # By default we work in a fail-fast mode
                run_cmd(cmd=cmd, echo=True, silent=False)
