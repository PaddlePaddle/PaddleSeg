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


import hashlib
from pathlib import Path

models_dir = Path()
ext = ".pdparams"
for model_path in models_dir.glob("*/*" + ext):
    md5 = hashlib.md5(model_path.read_bytes()).hexdigest()
    md5_path = str(model_path)[: -len(ext)] + ".md5"
    Path(md5_path).write_text(md5)
