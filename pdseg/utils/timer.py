# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import time


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step / speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
    return result.format(*arr)


class Timer(object):
    """ Simple timer class for measuring time consuming """

    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._elapsed_time = 0.0
        self._is_running = False

    def start(self):
        self._is_running = True
        self._start_time = time.time()

    def restart(self):
        self.start()

    def stop(self):
        self._is_running = False
        self._end_time = time.time()

    def elapsed_time(self):
        self._end_time = time.time()
        self._elapsed_time = self._end_time - self._start_time
        if not self.is_running:
            return 0.0

        return self._elapsed_time

    @property
    def is_running(self):
        return self._is_running
