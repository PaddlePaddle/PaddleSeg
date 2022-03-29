import inspect
from collections.abc import Sequence


class ComponentManager:
    def __init__(self, name=None):
        self._components_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._components_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self):
                raise KeyError(f"指定的下标 {item} 在长度为 {len(self)} 的 {self} 中越界")
            return list(self._components_dict.values())[item]
        if item not in self._components_dict.keys():
            raise KeyError(f"{self} 中不存在 {item}")
        return self._components_dict[item]

    def __iter__(self):
        for val in self._components_dict.values():
            yield val

    def keys(self):
        return list(self._components_dict.keys())

    def idx(self, item):
        for idx, val in enumerate(self.keys()):
            if val == item:
                return idx
        raise KeyError(f"{item} is not in {self}")

    @property
    def components_dict(self):
        return self._components_dict

    @property
    def name(self):
        return self._name

    def _add_single_component(self, component):
        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError("Expect class/function type, but received {}".
                            format(type(component)))

        # Obtain the internal name of the component
        component_name = component.__name__

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            raise KeyError("{} exists already!".format(component_name))
        else:
            # Take the internal name of the component as its key
            self._components_dict[component_name] = component

    def add_component(self, components):
        # Check whether the type is a sequence
        if isinstance(components, Sequence):
            for component in components:
                self._add_single_component(component)
        else:
            component = components
            self._add_single_component(component)

        return components


MODELS = ComponentManager("models")
ACTIONS = ComponentManager("actions")
