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

import os.path as osp
import contextlib
import tempfile

from .primitive import Primitive


class CopyFile(Primitive):
    def __init__(self, src, dst=None, comm_key=None):
        super().__init__(comm_key)
        self.src = src
        self.dst = dst

    def do(self, comm=None):
        # Not implemented yet
        raise NotImplementedError

    def set(self, comm=None):
        ctx = self._create_context(comm=comm)
        return ctx

    @contextlib.contextmanager
    def _create_context(self, comm):
        if self.dst is None:
            with tempfile.TemporaryDirectory() as td:
                dst = osp.join(td, osp.basename(self.src))
                if comm is not None:
                    comm[self.comm_key] = dst
                self._copy_file_content_to(self.src, dst)
                yield comm
        else:
            self._copy_file_content_to(self.src, self.dst)
            yield comm

    @staticmethod
    def _copy_file_content_to(src, dst):
        with open(src, 'r') as f:
            all_cont = f.read()
        with open(dst, 'w') as f:
            f.write(all_cont)
