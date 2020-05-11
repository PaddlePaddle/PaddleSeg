#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
import paddle.fluid as fluid
import os
import numpy as np
import time
import math
import yaml
import copy
import json
import utils.logging as logging
from collections import OrderedDict
from os import path as osp
from utils.pretrain_weights import get_pretrain_weights
import transforms.transforms as T
import utils
import __init__


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


class BaseAPI:
    def __init__(self):
        # 现有的CV模型都有这个属性，而这个属且也需要在eval时用到
        self.num_classes = None
        self.labels = None
        if __init__.env_info['place'] == 'cpu':
            self.places = fluid.cpu_places()
        else:
            self.places = fluid.cuda_places()
        self.exe = fluid.Executor(self.places[0])
        self.train_prog = None
        self.test_prog = None
        self.parallel_train_prog = None
        self.train_inputs = None
        self.test_inputs = None
        self.train_outputs = None
        self.test_outputs = None
        self.train_data_loader = None
        self.eval_metrics = None
        # 若模型是从inference model加载进来的，无法调用训练接口进行训练
        self.trainable = True
        # 是否使用多卡间同步BatchNorm均值和方差
        self.sync_bn = False
        # 当前模型状态
        self.status = 'Normal'

    def _get_single_card_bs(self, batch_size):
        if batch_size % len(self.places) == 0:
            return int(batch_size // len(self.places))
        else:
            raise Exception("Please support correct batch_size, \
                            which can be divided by available cards({}) in {}".
                            format(__init__.env_info['num'],
                                   __init__.env_info['place']))

    def build_program(self):
        # 构建训练网络
        self.train_inputs, self.train_outputs = self.build_net(mode='train')
        self.train_prog = fluid.default_main_program()
        startup_prog = fluid.default_startup_program()

        # 构建预测网络
        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, startup_prog):
            with fluid.unique_name.guard():
                self.test_inputs, self.test_outputs = self.build_net(
                    mode='test')
        self.test_prog = self.test_prog.clone(for_test=True)

    def arrange_transforms(self, transforms, mode='train'):
        # 给transforms添加arrange操作
        if transforms.transforms[-1].__class__.__name__.startswith('Arrange'):
            transforms.transforms[-1] = T.ArrangeSegmenter(mode=mode)
        else:
            transforms.transforms.append(T.ArrangeSegmenter(mode=mode))

    def build_train_data_loader(self, reader, batch_size):
        # 初始化data_loader
        if self.train_data_loader is None:
            self.train_data_loader = fluid.io.DataLoader.from_generator(
                feed_list=list(self.train_inputs.values()),
                capacity=64,
                use_double_buffer=True,
                iterable=True)
        batch_size_each_gpu = self._get_single_card_bs(batch_size)
        generator = reader.generator(
            batch_size=batch_size_each_gpu, drop_last=True)
        self.train_data_loader.set_sample_list_generator(
            reader.generator(batch_size=batch_size_each_gpu),
            places=self.places)

    def net_initialize(self,
                       startup_prog=None,
                       pretrain_weights=None,
                       fuse_bn=False,
                       save_dir='.',
                       sensitivities_file=None,
                       eval_metric_loss=0.05):
        if hasattr(self, 'backbone'):
            backbone = self.backbone
        else:
            backbone = self.__class__.__name__
        pretrain_weights = get_pretrain_weights(pretrain_weights, backbone,
                                                save_dir)
        if startup_prog is None:
            startup_prog = fluid.default_startup_program()
        self.exe.run(startup_prog)
        if pretrain_weights is not None:
            logging.info(
                "Load pretrain weights from {}.".format(pretrain_weights))
            utils.utils.load_pretrain_weights(self.exe, self.train_prog,
                                              pretrain_weights, fuse_bn)
        # 进行裁剪
        if sensitivities_file is not None:
            from .slim.prune_config import get_sensitivities
            sensitivities_file = get_sensitivities(sensitivities_file, self,
                                                   save_dir)
            from .slim.prune import get_params_ratios, prune_program
            prune_params_ratios = get_params_ratios(
                sensitivities_file, eval_metric_loss=eval_metric_loss)
            prune_program(self, prune_params_ratios)
            self.status = 'Prune'

    def get_model_info(self):
        info = dict()
        info['Model'] = self.__class__.__name__
        info['_Attributes'] = {}
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        info['_init_params'] = self.init_params

        info['_Attributes']['num_classes'] = self.num_classes
        info['_Attributes']['labels'] = self.labels
        try:
            primary_metric_key = list(self.eval_metrics.keys())[0]
            primary_metric_value = float(self.eval_metrics[primary_metric_key])
            info['_Attributes']['eval_metrics'] = {
                primary_metric_key: primary_metric_value
            }
        except:
            pass

        if hasattr(self, 'test_transforms'):
            if self.test_transforms is not None:
                info['Transforms'] = list()
                for op in self.test_transforms.transforms:
                    name = op.__class__.__name__
                    attr = op.__dict__
                    info['Transforms'].append({name: attr})
        return info

    def save_model(self, save_dir):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        fluid.save(self.train_prog, osp.join(save_dir, 'model'))
        model_info = self.get_model_info()
        model_info['status'] = self.status
        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)
        # 评估结果保存
        if hasattr(self, 'eval_details'):
            with open(osp.join(save_dir, 'eval_details.json'), 'w') as f:
                json.dump(self.eval_details, f)

        if self.status == 'Prune':
            # 保存裁剪的shape
            shapes = {}
            for block in self.train_prog.blocks:
                for param in block.all_parameters():
                    pd_var = fluid.global_scope().find_var(param.name)
                    pd_param = pd_var.get_tensor()
                    shapes[param.name] = np.array(pd_param).shape
            with open(
                    osp.join(save_dir, 'prune.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(shapes, f)

        # 模型保存成功的标志
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model saved in {}.".format(save_dir))

    def train_loop(self,
                   num_epochs,
                   train_reader,
                   train_batch_size,
                   eval_reader=None,
                   save_interval_epochs=1,
                   log_interval_steps=10,
                   save_dir='output',
                   use_vdl=False):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        if use_vdl:
            from visualdl import LogWriter
            vdl_logdir = osp.join(save_dir, 'vdl_log')
        # 给transform添加arrange操作
        self.arrange_transforms(
            transforms=train_reader.transforms, mode='train')
        # 构建train_data_loader
        self.build_train_data_loader(
            reader=train_reader, batch_size=train_batch_size)

        if eval_reader is not None:
            self.eval_transforms = eval_reader.transforms
            self.test_transforms = copy.deepcopy(eval_reader.transforms)

        # 获取实时变化的learning rate
        lr = self.optimizer._learning_rate
        if isinstance(lr, fluid.framework.Variable):
            self.train_outputs['lr'] = lr

        # 在多卡上跑训练
        if self.parallel_train_prog is None:
            build_strategy = fluid.compiler.BuildStrategy()
            build_strategy.fuse_all_optimizer_ops = False
            if __init__.env_info['place'] != 'cpu' and len(self.places) > 1:
                build_strategy.sync_batch_norm = self.sync_bn
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_iteration_per_drop_scope = 1
            self.parallel_train_prog = fluid.CompiledProgram(
                self.train_prog).with_data_parallel(
                    loss_name=self.train_outputs['loss'].name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)

        total_num_steps = math.floor(
            train_reader.num_samples / train_batch_size)
        num_steps = 0
        time_stat = list()

        if use_vdl:
            # VisualDL component
            log_writer = LogWriter(vdl_logdir, sync_cycle=20)
            train_step_component = OrderedDict()
            eval_component = OrderedDict()

        best_accuracy_key = ""
        best_accuracy = -1.0
        best_model_epoch = 1
        for i in range(num_epochs):
            records = list()
            step_start_time = time.time()
            for step, data in enumerate(self.train_data_loader()):
                outputs = self.exe.run(
                    self.parallel_train_prog,
                    feed=data,
                    fetch_list=list(self.train_outputs.values()))
                outputs_avg = np.mean(np.array(outputs), axis=1)
                records.append(outputs_avg)

                # 训练完成剩余时间预估
                current_time = time.time()
                step_cost_time = current_time - step_start_time
                step_start_time = current_time
                if len(time_stat) < 20:
                    time_stat.append(step_cost_time)
                else:
                    time_stat[num_steps % 20] = step_cost_time
                eta = ((num_epochs - i) * total_num_steps - step -
                       1) * np.mean(time_stat)
                eta_h = math.floor(eta / 3600)
                eta_m = math.floor((eta - eta_h * 3600) / 60)
                eta_s = int(eta - eta_h * 3600 - eta_m * 60)
                eta_str = "{}:{}:{}".format(eta_h, eta_m, eta_s)

                # 每间隔log_interval_steps，输出loss信息
                num_steps += 1
                if num_steps % log_interval_steps == 0:
                    step_metrics = OrderedDict(
                        zip(list(self.train_outputs.keys()), outputs_avg))

                    if use_vdl:
                        for k, v in step_metrics.items():
                            if k not in train_step_component.keys():
                                with log_writer.mode('Each_Step_while_Training'
                                                     ) as step_logger:
                                    train_step_component[
                                        k] = step_logger.scalar(
                                            'Training: {}'.format(k))
                            train_step_component[k].add_record(num_steps, v)

                    logging.info(
                        "[TRAIN] Epoch={}/{}, Step={}/{}, {}, eta={}".format(
                            i + 1, num_epochs, step + 1, total_num_steps,
                            dict2str(step_metrics), eta_str))
            train_metrics = OrderedDict(
                zip(list(self.train_outputs.keys()), np.mean(records, axis=0)))
            logging.info('[TRAIN] Epoch {} finished, {} .'.format(
                i + 1, dict2str(train_metrics)))

            # 每间隔save_interval_epochs, 在验证集上评估和对模型进行保存
            if (i + 1) % save_interval_epochs == 0 or i == num_epochs - 1:
                current_save_dir = osp.join(save_dir, "epoch_{}".format(i + 1))
                if not osp.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                if eval_reader is not None:
                    # 检测目前仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。
                    eval_batch_size = train_batch_size
                    self.eval_metrics, self.eval_details = self.evaluate(
                        eval_reader=eval_reader,
                        batch_size=eval_batch_size,
                        verbose=True,
                        epoch_id=i + 1,
                        return_details=True)
                    logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
                        i + 1, dict2str(self.eval_metrics)))
                    # 保存最优模型
                    best_accuracy_key = list(self.eval_metrics.keys())[0]
                    current_accuracy = self.eval_metrics[best_accuracy_key]
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_model_epoch = i + 1
                        best_model_dir = osp.join(save_dir, "best_model")
                        self.save_model(save_dir=best_model_dir)
                    if use_vdl:
                        for k, v in self.eval_metrics.items():
                            if isinstance(v, list):
                                continue
                            if isinstance(v, np.ndarray):
                                if v.size > 1:
                                    continue
                            if k not in eval_component:
                                with log_writer.mode('Each_Epoch_on_Eval_Data'
                                                     ) as eval_logger:
                                    eval_component[k] = eval_logger.scalar(
                                        'Evaluation: {}'.format(k))
                            eval_component[k].add_record(i + 1, v)
                self.save_model(save_dir=current_save_dir)
                logging.info(
                    'Current evaluated best model in eval_reader is epoch_{}, {}={}'
                    .format(best_model_epoch, best_accuracy_key, best_accuracy))
