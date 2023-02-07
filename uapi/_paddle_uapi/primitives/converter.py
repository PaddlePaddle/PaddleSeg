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


class Prim2FuncConverter(object):
    def __call__(self, primitive):
        return self.convert(primitive)

    def convert(self, primitive):
        # Since the eager operation is not supported by all primitives,
        # we wrap around the lazy operation.
        def _func(*args, comm=None, **kwargs):
            # TODO: Retain function signature
            nonlocal primitive
            prim = primitive(*args, **kwargs)
            if comm is None:
                comm = dict()
            with prim.set(comm=comm) as comm:
                pass
            # We do not ensure that the items in `comm` is not expired
            return comm

        return _func
