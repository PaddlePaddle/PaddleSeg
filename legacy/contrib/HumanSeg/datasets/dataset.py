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

import os.path as osp
from threading import Thread
import multiprocessing
import collections
import numpy as np
import six
import sys
import copy
import random
import platform
import chardet
import utils.logging as logging


class EndSignal():
    pass


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


def is_valid(sample):
    if sample is None:
        return False
    if isinstance(sample, tuple):
        for s in sample:
            if s is None:
                return False
            elif isinstance(s, np.ndarray) and s.size == 0:
                return False
            elif isinstance(s, collections.Sequence) and len(s) == 0:
                return False
    return True


def get_encoding(path):
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    return file_encoding


def multithread_reader(mapper,
                       reader,
                       num_workers=4,
                       buffer_size=1024,
                       batch_size=8,
                       drop_last=True):
    from queue import Queue
    end = EndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for i in reader():
            in_queue.put(i)
        in_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, EndSignal):
            if len(sample) == 2:
                r = mapper(sample[0], sample[1])
            elif len(sample) == 3:
                r = mapper(sample[0], sample[1], sample[2])
            else:
                raise Exception('The sample\'s length must be 2 or 3.')
            if is_valid(r):
                out_queue.put(r)
            sample = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    def xreader():
        in_queue = Queue(buffer_size)
        out_queue = Queue(buffer_size)
        # start a read worker in a thread
        target = read_worker
        t = Thread(target=target, args=(reader, in_queue))
        t.daemon = True
        t.start()
        # start several handle_workers
        target = handle_worker
        args = (in_queue, out_queue, mapper)
        workers = []
        for i in range(num_workers):
            worker = Thread(target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        batch_data = []
        sample = out_queue.get()
        while not isinstance(sample, EndSignal):
            batch_data.append(sample)
            if len(batch_data) == batch_size:
                yield batch_data
                batch_data = []
            sample = out_queue.get()
        finish = 1
        while finish < num_workers:
            sample = out_queue.get()
            if isinstance(sample, EndSignal):
                finish += 1
            else:
                batch_data.append(sample)
                if len(batch_data) == batch_size:
                    yield batch_data
                    batch_data = []
        if not drop_last and len(batch_data) != 0:
            yield batch_data
            batch_data = []

    return xreader


def multiprocess_reader(mapper,
                        reader,
                        num_workers=4,
                        buffer_size=1024,
                        batch_size=8,
                        drop_last=True):
    from .shared_queue import SharedQueue as Queue

    def _read_into_queue(samples, mapper, queue):
        end = EndSignal()
        try:
            for sample in samples:
                if sample is None:
                    raise ValueError("sample has None")
                if len(sample) == 2:
                    result = mapper(sample[0], sample[1])
                elif len(sample) == 3:
                    result = mapper(sample[0], sample[1], sample[2])
                else:
                    raise Exception('The sample\'s length must be 2 or 3.')
                if is_valid(result):
                    queue.put(result)
            queue.put(end)
        except:
            queue.put("")
            six.reraise(*sys.exc_info())

    def queue_reader():
        queue = Queue(buffer_size, memsize=3 * 1024**3)
        total_samples = [[] for i in range(num_workers)]
        for i, sample in enumerate(reader()):
            index = i % num_workers
            total_samples[index].append(sample)
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=_read_into_queue, args=(total_samples[i], mapper, queue))
            p.start()

        finish_num = 0
        batch_data = list()
        while finish_num < num_workers:
            sample = queue.get()
            if isinstance(sample, EndSignal):
                finish_num += 1
            elif sample == "":
                raise ValueError("multiprocess reader raises an exception")
            else:
                batch_data.append(sample)
                if len(batch_data) == batch_size:
                    yield batch_data
                    batch_data = []
        if len(batch_data) != 0 and not drop_last:
            yield batch_data
            batch_data = []

    return queue_reader


class Dataset:
    def __init__(self,
                 data_dir,
                 file_list,
                 label_list=None,
                 transforms=None,
                 num_workers='auto',
                 buffer_size=100,
                 parallel_method='thread',
                 shuffle=False):
        if num_workers == 'auto':
            import multiprocessing as mp
            num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 8 else 8
        if transforms is None:
            raise Exception("transform should be defined.")
        self.transforms = transforms
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.parallel_method = parallel_method
        self.shuffle = shuffle

        self.file_list = list()
        self.labels = list()
        self._epoch = 0

        if label_list is not None:
            with open(label_list, encoding=get_encoding(label_list)) as f:
                for line in f:
                    item = line.strip()
                    self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                items = line.strip().split()
                if not is_pic(items[0]):
                    continue
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not osp.exists(full_path_im):
                    raise IOError(
                        'The image file {} is not exist!'.format(full_path_im))
                if not osp.exists(full_path_label):
                    raise IOError('The image file {} is not exist!'.format(
                        full_path_label))
                self.file_list.append([full_path_im, full_path_label])
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def iterator(self):
        self._epoch += 1
        self._pos = 0
        files = copy.deepcopy(self.file_list)
        if self.shuffle:
            random.shuffle(files)
        files = files[:self.num_samples]
        self.num_samples = len(files)
        for f in files:
            label_path = f[1]
            sample = [f[0], None, label_path]
            yield sample

    def generator(self, batch_size=1, drop_last=True):
        self.batch_size = batch_size
        parallel_reader = multithread_reader
        if self.parallel_method == "process":
            if platform.platform().startswith("Windows"):
                logging.debug(
                    "multiprocess_reader is not supported in Windows platform, force to use multithread_reader."
                )
            else:
                parallel_reader = multiprocess_reader
        return parallel_reader(
            self.transforms,
            self.iterator,
            num_workers=self.num_workers,
            buffer_size=self.buffer_size,
            batch_size=batch_size,
            drop_last=drop_last)
