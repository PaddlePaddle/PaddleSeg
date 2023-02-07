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

import contextlib

from .primitive import Primitive
from ..arg import CLIArgument


class AppendCLIArg(Primitive):
    def __init__(self, key, val, sep=' ', comm_key=None):
        super().__init__(comm_key)
        self.arg = CLIArgument(key=key, val=val, sep=sep)

    def do(self, comm):
        # Not implemented yet
        raise NotImplementedError

    def set(self, comm):
        ctx = self._create_context(comm=comm)
        return ctx

    @contextlib.contextmanager
    def _create_context(self, comm):
        comm[self.comm_key].append(self.arg)
        yield comm
