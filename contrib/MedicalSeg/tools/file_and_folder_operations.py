# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import os
import pickle
import json
from typing import List


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def pardir(path: str):
    return os.path.join(path, os.pardir)


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)


# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split

# I am tired of confusing those
subfolders = subdirs
save_pickle = write_pickle
write_json = save_json
