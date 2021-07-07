from functools import wraps
from copy import deepcopy
import inspect
import paddle.nn as nn



def serialize(init):
    parameters = list(inspect.signature(init).parameters)

    @wraps(init)
    def new_init(self, *args, **kwargs):
        params = deepcopy(kwargs)
        for pname, value in zip(parameters[1:], args):
            params[pname] = value

        config = {
            'class': get_classname(self.__class__),
            'params': dict()
        }
        specified_params = set(params.keys())

        for pname, param in get_default_params(self.__class__).items():
            if pname not in params:
                params[pname] = param.default

        for name, value in list(params.items()):
            param_type = 'builtin'
            if inspect.isclass(value):
                param_type = 'class'
                value = get_classname(value)

            config['params'][name] = {
                'type': param_type,
                'value': value,
                'specified': name in specified_params
            }

        setattr(self, '_config', config)
        init(self, *args, **kwargs)

    return new_init


def load_model(config, **kwargs):
    model_class = get_class_from_str(config['class'])
    model_default_params = get_default_params(model_class)

    model_args = dict()
    for pname, param in config['params'].items():
        value = param['value']
        if param['type'] == 'class':
            value = get_class_from_str(value)

        if pname not in model_default_params and not param['specified']:
            continue

        assert pname in model_default_params
        if not param['specified'] and model_default_params[pname].default == value:
            continue
        model_args[pname] = value

    model_args.update(kwargs)

    return model_class(**model_args)


def get_config_repr(config):
    config_str = f'Model: {config["class"]}\n'
    for pname, param in config['params'].items():
        value = param["value"]
        if param['type'] == 'class':
            value = value.split('.')[-1]
        param_str = f'{pname:<22} = {str(value):<12}'
        if not param['specified']:
            param_str += ' (default)'
        config_str += param_str + '\n'
    return config_str


def get_default_params(some_class):
    params = dict()
    for mclass in some_class.mro():
        if mclass is nn.Layer or mclass is object:
            continue

        mclass_params = inspect.signature(mclass.__init__).parameters
        for pname, param in mclass_params.items():
            if param.default != param.empty and pname not in params:
                params[pname] = param

    return params


def get_classname(cls):
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


def get_class_from_str(class_str):
    components = class_str.split('.')
    mod = __import__('.'.join(components[:-1]))
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
