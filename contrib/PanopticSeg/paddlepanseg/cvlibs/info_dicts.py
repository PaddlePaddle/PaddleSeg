# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from itertools import cycle
from collections import OrderedDict
from collections.abc import MutableMapping

import paddle


class _Constant(object):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __get__(self, obj, type_=None):
        return self.val

    def __set__(self, obj, val):
        raise AttributeError("The value of a constant cannot be modified!")


class InfoDict(MutableMapping):
    PRIMARY_KEYS = ()
    _DICT_TYPE = _Constant(OrderedDict)

    def __init__(self, *args, **kwargs):
        self._dict = OrderedDict(zip(self.PRIMARY_KEYS, cycle([None])))
        self.update(OrderedDict(*args, **kwargs))

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def collect(self, keys=None, return_dict=False):
        if keys is None:
            keys = self.PRIMARY_KEYS
        vals = map(self.__getitem__, keys)
        if return_dict:
            return self._DICT_TYPE(zip(keys, vals))
        else:
            return list(vals)

    def prune(self):
        keys = [k for k in self.keys() if self[k] is None]
        for key in keys:
            self._dict.pop(key)

    def __str__(self):
        return str(self._dict)


class SampleDict(InfoDict):
    PRIMARY_KEYS = ('img', 'label', 'gt_fields', 'img_path', 'lab_path', 'ann',
                    'pan_label', 'sem_label', 'ins_label', 'trans_info',
                    'image_id')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._dict['gt_fields'] is None:
            self._dict['gt_fields'] = [
                'label', 'pan_label', 'sem_label', 'ins_label'
            ]
        # XXX: Rewrite trans_info as empty list
        if self._dict['trans_info'] is None:
            self._dict['trans_info'] = []

    def __getitem__(self, key):
        if key == 'gt_fields':
            return [
                field for field in self._dict['gt_fields']
                if self.get(field, None) is not None
            ]
        return super().__getitem__(key)

    def prune(self):
        super().prune()
        for field in self._dict['gt_fields']:
            if field not in self:
                self._dict['gt_fields'].remove(field)


class NetOutDict(InfoDict):
    PRIMARY_KEYS = ('sem_out', 'ins_out', 'map_fields')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._dict['map_fields'] is None:
            self._dict['map_fields'] = ['sem_out']

    def prune(self):
        super().prune()
        for field in self._dict['map_fields']:
            if field not in self:
                self._dict['map_fields'].remove(field)


class PPOutDict(InfoDict):
    PRIMARY_KEYS = ('pan_pred', 'sem_pred', 'ins_pred')


class MetricDict(InfoDict):
    PRIMARY_KEYS = ('pan_metrics', 'sem_metrics', 'ins_metrics')


def build_info_dict(_type_, *args, **kwargs):
    type_ = _type_.lower()
    if type_ == 'sample':
        dict_ = SampleDict(*args, **kwargs)
    elif type_ == 'net_out':
        dict_ = NetOutDict(*args, **kwargs)
        if paddle.distributed.ParallelEnv().nranks > 1:
            dict_ = dict(dict_)
    elif type_ == 'pp_out':
        dict_ = PPOutDict(*args, **kwargs)
    elif type_ == 'metric':
        dict_ = MetricDict(*args, **kwargs)
    else:
        raise ValueError(f"{_type_} is not a supported info dict type.")
    return dict_
