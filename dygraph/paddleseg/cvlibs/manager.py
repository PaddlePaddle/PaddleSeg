# -*- encoding: utf-8 -*-
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

from collections.abc import Sequence
import inspect


class ComponentManager:
    """
    Implement a manager class to add the new component properly.
    The component can be added as either class or function type.
    For example:
        >>> model_manager = ComponentManager()
        >>> class AlexNet: ...
        >>> class ResNet: ...
        >>> model_manager.add_component(AlexNet)
        >>> model_manager.add_component(ResNet)
        or pass a sequence alliteratively:
        >>> model_manager.add_component([AlexNet, ResNet])
        >>> print(model_manager.components_dict)
    output: {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Or an easier way, using it as a Python decorator, while just add it above the class declaration.
        >>> model_manager = ComponentManager()
        >>> @model_manager.add_component
        >>> class AlexNet: ...
        >>> @model_manager.add_component
        >>> class ResNet: ...
        >>> print(model_manager.components_dict)
    output: {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
    """

    def __init__(self):
        self._components_dict = dict()

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        return "{}:{}".format(self.__class__.__name__,
                              list(self._components_dict.keys()))

    def __getitem__(self, item):
        if item not in self._components_dict.keys():
            raise KeyError("{} does not exist in the current {}".format(
                item, self))
        return self._components_dict[item]

    @property
    def components_dict(self):
        return self._components_dict

    def _add_single_component(self, component):
        """
        Add a single component into the corresponding manager

        Args:
        component (function | class): a new component

        Returns:
        None
        """

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError(
                "Expect class/function type, but received {}".format(
                    type(component)))

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            raise KeyError("{} exists already!".format(component_name))
        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(self, components):
        """
        Add component(s) into the corresponding manager

        Args:
        components (function | class | list | tuple): support three types of components

        Returns:
        None
        """

        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


MODELS = ComponentManager()
BACKBONES = ComponentManager()
DATASETS = ComponentManager()
TRANSFORMS = ComponentManager()
LOSSES = ComponentManager()
