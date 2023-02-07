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


class AppendToFile(Primitive):
    def __init__(self, content, file_path=None, comm_key=None):
        super().__init__(comm_key)
        self.content = content
        self.file_path = file_path

    def do(self, comm=None):
        # Not implemented yet
        raise NotImplementedError

    def set(self, comm=None):
        ctx = self._create_context(comm=comm)
        return ctx

    @contextlib.contextmanager
    def _create_context(self, comm):
        if self.file_path is None:
            file_path = comm[self.comm_key]
        else:
            file_path = self.file_path
        with open(file_path, 'a') as f:
            f.write(self.content)
        yield comm
