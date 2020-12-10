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

import os
import time

import numpy as np
import paddle
from paddle.distributed.parallel import ParallelEnv
from visualdl import LogWriter
from paddleseg.utils.progbar import Progbar
import paddleseg.utils.logger as logger


class CallbackList(object):
    """
    Container abstracting a list of callbacks.

    Args:
        callbacks (list[Callback]): List of `Callback` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, optimizer):
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def on_iter_begin(self, iter, logs=None):
        """Called right before processing a batch.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_iter_begin(iter, logs)
        self._t_enter_iter = time.time()

    def on_iter_end(self, iter, logs=None):
        """Called at the end of a batch.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_iter_end(iter, logs)
        self._t_exit_iter = time.time()

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.validation_data = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_iter_begin(self, iter, logs=None):
        pass

    def on_iter_end(self, iter, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class BaseLogger(Callback):
    def __init__(self, period=10):
        super(BaseLogger, self).__init__()
        self.period = period

    def _reset(self):
        self.totals = {}

    def on_train_begin(self, logs=None):
        self.totals = {}

    def on_iter_end(self, iter, logs=None):
        logs = logs or {}
        #(iter - 1) // iters_per_epoch + 1
        for k, v in logs.items():
            if k in self.totals.keys():
                self.totals[k] += v
            else:
                self.totals[k] = v

        if iter % self.period == 0 and ParallelEnv().local_rank == 0:

            for k in self.totals:
                logs[k] = self.totals[k] / self.period
            self._reset()


class TrainLogger(Callback):
    def __init__(self, log_freq=10):
        self.log_freq = log_freq

    def _calculate_eta(self, remaining_iters, speed):
        if remaining_iters < 0:
            remaining_iters = 0
        remaining_time = int(remaining_iters * speed)
        result = "{:0>2}:{:0>2}:{:0>2}"
        arr = []
        for i in range(2, -1, -1):
            arr.append(int(remaining_time / 60**i))
            remaining_time %= 60**i
        return result.format(*arr)

    def on_iter_end(self, iter, logs=None):

        if iter % self.log_freq == 0 and ParallelEnv().local_rank == 0:
            total_iters = self.params["total_iters"]
            iters_per_epoch = self.params["iters_per_epoch"]
            remaining_iters = total_iters - iter
            eta = self._calculate_eta(remaining_iters, logs["batch_cost"])
            current_epoch = (iter - 1) // self.params["iters_per_epoch"] + 1
            loss = logs["loss"]
            lr = self.optimizer.get_lr()
            batch_cost = logs["batch_cost"]
            reader_cost = logs["reader_cost"]

            logger.info(
                "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.4f} | ETA {}"
                .format(current_epoch, iter, total_iters, loss, lr, batch_cost,
                        reader_cost, eta))


class ProgbarLogger(Callback):
    def __init__(self):
        super(ProgbarLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.verbose = self.params["verbose"]
        self.total_iters = self.params["total_iters"]
        self.target = self.params["total_iters"]
        self.progbar = Progbar(target=self.target, verbose=self.verbose)
        self.seen = 0
        self.log_values = []

    def on_iter_begin(self, iter, logs=None):
        #self.seen = 0
        if self.seen < self.target:
            self.log_values = []

    def on_iter_end(self, iter, logs=None):
        logs = logs or {}
        self.seen += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        #if self.verbose and self.seen < self.target and ParallelEnv.local_rank == 0:
        #print(self.log_values)
        if self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)


class ModelCheckpoint(Callback):
    def __init__(self,
                 save_dir,
                 monitor="miou",
                 save_best_only=False,
                 save_params_only=True,
                 mode="max",
                 period=1):

        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.save_params_only = save_params_only
        self.period = period
        self.iters_since_last_save = 0

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise RuntimeError("`mode` is neither \"min\" nor \"max\"!")

    def on_train_begin(self, logs=None):
        self.verbose = self.params["verbose"]
        save_dir = self.save_dir
        if not os.path.isdir(save_dir):
            if os.path.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)

    def on_iter_end(self, iter, logs=None):
        logs = logs or {}
        self.iters_since_last_save += 1
        current_save_dir = os.path.join(self.save_dir, "iter_{}".format(iter))
        current_save_dir = os.path.abspath(current_save_dir)
        #if self.iters_since_last_save % self.period and ParallelEnv().local_rank == 0:
        #self.iters_since_last_save = 0
        if iter % self.period == 0 and ParallelEnv().local_rank == 0:
            if self.verbose > 0:
                print("iter {iter_num}: saving model to {path}".format(
                    iter_num=iter, path=current_save_dir))

            paddle.save(self.model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))

            if not self.save_params_only:
                paddle.save(self.optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))


class VisualDL(Callback):
    def __init__(self, log_dir="./log", freq=1):
        super(VisualDL, self).__init__()
        self.log_dir = log_dir
        self.freq = freq

    def on_train_begin(self, logs=None):
        self.writer = LogWriter(self.log_dir)

    def on_iter_end(self, iter, logs=None):
        logs = logs or {}
        if iter % self.freq == 0 and ParallelEnv().local_rank == 0:
            for k, v in logs.items():
                self.writer.add_scalar("Train/{}".format(k), v, iter)

        self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()
