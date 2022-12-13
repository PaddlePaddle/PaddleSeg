# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import os.path as osp
import sys
import contextlib
import itertools

import paddlepanseg.apis as api

PROJ_DIR = os.environ.get('PADDLEPANSEG_PROJ_DIR', None)
if PROJ_DIR is None:
    # XXX: Relative position of default project dir to current file is hard-coded here
    PROJ_DIR = osp.join(osp.dirname(__file__), '..', 'projects')
PROJ_LIST = os.listdir(PROJ_DIR)

_PROJ_INIT_STATES = dict(zip(PROJ_LIST, itertools.cycle([False])))


def _get_proj_path(proj_name):
    proj_path = osp.join(PROJ_DIR, proj_name)
    return proj_path


def init_project(proj_name):
    global PROJ_LIST, _PROJ_INIT_STATES
    if proj_name not in PROJ_LIST:
        raise ValueError(f"Cannot find the project {proj_name}.")
    if _PROJ_INIT_STATES[proj_name]:
        return
    proj_path = _get_proj_path(proj_name)
    # O(N) linear search
    if proj_path not in sys.path:
        # Add `proj_path` to module search path
        sys.path.append(proj_path)
    # Load all modules in the project into memory
    # By doing so all components should be registered
    api.load_modules()
    _PROJ_INIT_STATES[proj_name] = True


def init_all_projects():
    for proj in PROJ_LIST:
        init_project(proj)


@contextlib.contextmanager
def work_on_project(proj_name, switch_dir=False):
    init_project(proj_name)
    if switch_dir:
        proj_path = _get_proj_path(proj_name)
        cwd = os.getcwd()
        os.chdir(proj_path)
    try:
        yield
    finally:
        if switch_dir:
            os.chdir(cwd)
