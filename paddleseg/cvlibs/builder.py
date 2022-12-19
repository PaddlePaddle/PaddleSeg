# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class ComponentBuilder(object):
    """
    This class is responsible for building components. All component classes must be available 
        in the list of maintained components.

    Args:
        com_list (list): A list of component classes.
    """

    def __init__(self, com_list):
        super().__init__()
        self.com_list = com_list

    def load_component_class(self, com_name):
        """
        Load component class, such as model, loss, dataset, etc.
        """
        for com in self.com_list:
            if com_name in com.components_dict:
                return com[com_name]

        raise RuntimeError("The specified component ({}) was not found.".format(
            com_name))

    def create_object(self, cfg):
        """
        Create Python object, such as model, loss, dataset, etc.
        """
        component = self.load_component_class_from_config(cfg)

        params = {}
        for key, val in cfg.items():
            if self.is_meta_type(val):
                params[key] = self.create_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self.create_object(item)
                    if self.is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val

        return self._create_object_impl(component, **params)

    def _create_object_impl(self, component, *args, **kwargs):
        raise NotImplementedError

    def load_component_class_from_config(self, cfg):
        raise NotImplementedError

    @classmethod
    def is_meta_type(cls, obj):
        raise NotImplementedError


class DefaultComponentBuilder(ComponentBuilder):
    def _create_object_impl(self, component, *args, **kwargs):
        try:
            return component(*args, **kwargs)
        except Exception as e:
            if hasattr(component, '__name__'):
                com_name = component.__name__
            else:
                com_name = ''
            raise RuntimeError(
                f"Tried to create a {com_name} object, but the operation has failed. "
                "Please double check the arguments used to create the object.\n"
                f"The error message is: \n{str(e)}")

    def load_component_class_from_config(self, cfg):
        # XXX: `cfg` is modified in place. We pop out the 'type' key.
        if 'type' not in cfg:
            raise RuntimeError(
                "It is not possible to create a component object from {}, as 'type' is not specified.".
                format(cfg))
        component = self.load_component_class(cfg.pop('type'))
        return component

    @classmethod
    def is_meta_type(cls, obj):
        # TODO: should we define a protocol (see https://peps.python.org/pep-0544/#defining-a-protocol)
        # to make it more pythonic?
        return isinstance(obj, dict) and 'type' in obj
