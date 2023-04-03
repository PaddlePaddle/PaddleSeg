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

__all__ = ['register', 'create']

global_config = dict()


def register(cls):
    """
    Register a given module class.
    Args:
        cls (type): Module class to be registered.
    Returns: cls
    """
    if cls.__name__ in global_config:
        raise ValueError("Module class already registered: {}".format(
            cls.__name__))
    global_config[cls.__name__] = cls
    return cls


def create(cls_name, model_cfg, env_cfg=None):
    """
    Create an instance of given module class.

    Args:
        cls_name(str): Class of which to create instnce.

    Return: instance of type `cls_or_name`
    """
    assert type(cls_name) == str, "should be a name of class"
    if cls_name not in global_config:
        raise ValueError("The module {} is not registered".format(cls_name))
    cls = global_config[cls_name]
    return cls(model_cfg, env_cfg)


def get_global_op():
    return global_config
