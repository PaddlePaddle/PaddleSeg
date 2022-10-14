# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import paddle

import traceback
from multiprocessing import Event, Queue, Process
from time import sleep
from threadpoolctl import threadpool_limits


def prefetch_data(queue,
                  data_loader,
                  transform,
                  queue_min_size,
                  process_id,
                  wait_time=0.2):
    while True:
        try:
            cur_size = queue.qsize()
            if cur_size < queue_min_size:
                item = next(data_loader)
                item = transform(**item)
                if not queue.full():
                    queue.put(item)
            else:
                sleep(wait_time)
        except Exception as exp:
            raise exp
    return


class MultiThreadedAugmenter:
    def __init__(self,
                 data_loader,
                 transform,
                 num_processes,
                 num_cached_per_queue=2,
                 timeout=10,
                 wait_time=0.2):
        self.data_loader = data_loader
        self.transform = transform
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self.queue_maxsize = num_cached_per_queue * num_processes
        self.timeout = timeout
        self.wait_time = wait_time
        self._queue = None
        self._processes = None
        self.is_initialized = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if not self.is_initialized:
            self.initialize()
            self.is_initialized = True
        try:
            while self._queue.empty():
                sleep(self.wait_time)
            item = self._queue.get(timeout=self.timeout)
        except KeyboardInterrupt:
            print("MultiThreadedGenerator: caught exception: {}".format(
                sys.exc_info()))
            self._finish()
        except Exception as e:
            print("Exception: {}".format(e))
            self._finish()
        return item

    def initialize(self):
        self._queue = Queue(maxsize=self.queue_maxsize)
        self._processes = [
            Process(
                target=prefetch_data,
                name="MultiThreadedAugmenter_{}".format(i),
                args=(self._queue, self.data_loader, self.transform,
                      self.queue_maxsize / 2, i, self.wait_time))
            for i in range(self.num_processes)
        ]
        for proc in self._processes:
            proc.daemon = True
            proc.start()

    def _finish(self):
        if self.is_initialized:
            for proc in self._processes:
                if proc.is_alive():
                    proc.terminate()
            self._queue.close()
            self._queue.join_thread()
            del self._queue
            del self._processes

    def __del__(self):
        self._finish()
