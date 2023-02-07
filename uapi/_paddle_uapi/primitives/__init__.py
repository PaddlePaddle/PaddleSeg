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

from .append_cli_arg import AppendCLIArg
from .copy_file import CopyFile
from .modify_yaml_cfg_val import ModifyYAMLCfgVal
from .communicate import Communicate
from .append_to_file import AppendToFile

# Build functional version of primitives
from .converter import Prim2FuncConverter
_cvrter = Prim2FuncConverter()
do_append_cli_arg = _cvrter(AppendCLIArg)
do_copy_file = _cvrter(CopyFile)
do_modify_yaml_cfg_val = _cvrter(ModifyYAMLCfgVal)
do_append_to_file = _cvrter(AppendToFile)
