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

from __future__ import absolute_import
import paddle.fluid as fluid
import os
from os import path as osp
import numpy as np
from collections import OrderedDict
import copy
import math
import time
import tqdm
import cv2
import yaml
import shutil
import paddleslim as slim
import paddle

import utils
import utils.logging as logging
from utils import seconds_to_hms
from utils import ConfusionMatrix
from utils import get_environ_info
from nets import DeepLabv3p, ShuffleSeg, HRNet
import transforms as T


def save_infer_program(test_program, ckpt_dir):
    _test_program = test_program.clone()
    _test_program.desc.flush()
    _test_program.desc._set_version()
    paddle.fluid.core.save_op_compatible_info(_test_program.desc)
    with open(os.path.join(ckpt_dir, 'model') + ".pdmodel", "wb") as f:
        f.write(_test_program.desc.serialize_to_string())


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


class SegModel(object):
    # DeepLab mobilenet
    def __init__(self,
                 num_classes=2,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 sync_bn=True):
        self.init_params = locals()
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        if class_weight is not None:
            if isinstance(class_weight, list):
                if len(class_weight) != num_classes:
                    raise ValueError(
                        "Length of class_weight should be equal to number of classes"
                    )
            elif isinstance(class_weight, str):
                if class_weight.lower() != 'dynamic':
                    raise ValueError(
                        "if class_weight is string, must be dynamic!")
            else:
                raise TypeError(
                    'Expect class_weight is a list or string but receive {}'.
                    format(type(class_weight)))

        self.num_classes = num_classes
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.sync_bn = sync_bn

        self.labels = None
        self.env_info = get_environ_info()
        if self.env_info['place'] == 'cpu':
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
        # 当前模型状态
        self.status = 'Normal'

    def _get_single_card_bs(self, batch_size):
        if batch_size % len(self.places) == 0:
            return int(batch_size // len(self.places))
        else:
            raise Exception("Please support correct batch_size, \
                            which can be divided by available cards({}) in {}".
                            format(self.env_info['num'],
                                   self.env_info['place']))

    def build_net(self, mode='train'):
        """应根据不同的情况进行构建"""
        pass

    def build_program(self):
        # build training network
        self.train_inputs, self.train_outputs = self.build_net(mode='train')
        self.train_prog = fluid.default_main_program()
        startup_prog = fluid.default_startup_program()

        # build prediction network
        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, startup_prog):
            with fluid.unique_name.guard():
                self.test_inputs, self.test_outputs = self.build_net(
                    mode='test')
        self.test_prog = self.test_prog.clone(for_test=True)

    def arrange_transform(self, transforms, mode='train'):
        arrange_transform = T.ArrangeSegmenter
        if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
            transforms.transforms[-1] = arrange_transform(mode=mode)
        else:
            transforms.transforms.append(arrange_transform(mode=mode))

    def build_train_data_loader(self, dataset, batch_size):
        # init data_loader
        if self.train_data_loader is None:
            self.train_data_loader = fluid.io.DataLoader.from_generator(
                feed_list=list(self.train_inputs.values()),
                capacity=64,
                use_double_buffer=True,
                iterable=True)
        batch_size_each_gpu = self._get_single_card_bs(batch_size)
        self.train_data_loader.set_sample_list_generator(
            dataset.generator(batch_size=batch_size_each_gpu),
            places=self.places)

    def net_initialize(self,
                       startup_prog=None,
                       pretrained_weights=None,
                       resume_weights=None):
        if startup_prog is None:
            startup_prog = fluid.default_startup_program()
        self.exe.run(startup_prog)
        if resume_weights is not None:
            logging.info("Resume weights from {}".format(resume_weights))
            if not osp.exists(resume_weights):
                raise Exception("Path {} not exists.".format(resume_weights))
            fluid.load(self.train_prog, osp.join(resume_weights, 'model'),
                       self.exe)
            # Check is path ended by path spearator
            if resume_weights[-1] == os.sep:
                resume_weights = resume_weights[0:-1]
            epoch_name = osp.basename(resume_weights)
            # If resume weights is end of digit, restore epoch status
            epoch = epoch_name.split('_')[-1]
            if epoch.isdigit():
                self.begin_epoch = int(epoch)
            else:
                raise ValueError("Resume model path is not valid!")
            logging.info("Model checkpoint loaded successfully!")

        elif pretrained_weights is not None:
            logging.info(
                "Load pretrain weights from {}.".format(pretrained_weights))
            utils.load_pretrained_weights(self.exe, self.train_prog,
                                          pretrained_weights)

    def get_model_info(self):
        # 存储相应的信息到yml文件
        info = dict()
        info['Model'] = self.__class__.__name__
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        info['_init_params'] = self.init_params

        info['_Attributes'] = dict()
        info['_Attributes']['num_classes'] = self.num_classes
        info['_Attributes']['labels'] = self.labels
        try:
            info['_Attributes']['eval_metric'] = dict()
            for k, v in self.eval_metrics.items():
                if isinstance(v, np.ndarray):
                    if v.size > 1:
                        v = [float(i) for i in v]
                else:
                    v = float(v)
                info['_Attributes']['eval_metric'][k] = v
        except:
            pass

        if hasattr(self, 'test_transforms'):
            if self.test_transforms is not None:
                info['test_transforms'] = list()
                for op in self.test_transforms.transforms:
                    name = op.__class__.__name__
                    attr = op.__dict__
                    info['test_transforms'].append({name: attr})

        if hasattr(self, 'train_transforms'):
            if self.train_transforms is not None:
                info['train_transforms'] = list()
                for op in self.train_transforms.transforms:
                    name = op.__class__.__name__
                    attr = op.__dict__
                    info['train_transforms'].append({name: attr})

        if hasattr(self, 'train_init'):
            if 'self' in self.train_init:
                del self.train_init['self']
            if 'train_dataset' in self.train_init:
                del self.train_init['train_dataset']
            if 'eval_dataset' in self.train_init:
                del self.train_init['eval_dataset']
            if 'optimizer' in self.train_init:
                del self.train_init['optimizer']
            info['train_init'] = self.train_init
        return info

    def save_model(self, save_dir):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        model_info = self.get_model_info()

        if self.status == 'Normal':
            fluid.save(self.train_prog, osp.join(save_dir, 'model'))
            save_infer_program(self.test_prog, save_dir)
            model_info['status'] = 'Normal'
        elif self.status == 'Quant':
            fluid.save(self.test_prog, osp.join(save_dir, 'model'))
            model_info['status'] = 'QuantOnline'

        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        # The flag of model for saving successfully
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model saved in {}.".format(save_dir))

    def export_inference_model(self, save_dir):
        test_input_names = [var.name for var in list(self.test_inputs.values())]
        test_outputs = list(self.test_outputs.values())
        fluid.io.save_inference_model(
            dirname=save_dir,
            executor=self.exe,
            params_filename='__params__',
            feeded_var_names=test_input_names,
            target_vars=test_outputs,
            main_program=self.test_prog)
        model_info = self.get_model_info()
        model_info['status'] = 'Infer'

        # Save input and output descrition of model
        model_info['_ModelInputsOutputs'] = dict()
        model_info['_ModelInputsOutputs']['test_inputs'] = [
            [k, v.name] for k, v in self.test_inputs.items()
        ]
        model_info['_ModelInputsOutputs']['test_outputs'] = [
            [k, v.name] for k, v in self.test_outputs.items()
        ]

        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        # The flag of model for saving successfully
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model for inference deploy saved in {}.".format(save_dir))

    def export_quant_model(self,
                           dataset=None,
                           save_dir=None,
                           batch_size=1,
                           batch_nums=10,
                           cache_dir=".temp",
                           quant_type="offline"):
        if quant_type == "offline":
            self.arrange_transform(transforms=dataset.transforms, mode='quant')
            dataset.num_samples = batch_size * batch_nums
            try:
                from utils import HumanSegPostTrainingQuantization
            except:
                raise Exception(
                    "Model Quantization is not available, try to upgrade your paddlepaddle>=1.8.1"
                )
            is_use_cache_file = True
            if cache_dir is None:
                is_use_cache_file = False
            post_training_quantization = HumanSegPostTrainingQuantization(
                executor=self.exe,
                dataset=dataset,
                program=self.test_prog,
                inputs=self.test_inputs,
                outputs=self.test_outputs,
                batch_size=batch_size,
                batch_nums=batch_nums,
                scope=None,
                algo='KL',
                quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                is_full_quantize=False,
                is_use_cache_file=is_use_cache_file,
                cache_dir=cache_dir)
            post_training_quantization.quantize()
            post_training_quantization.save_quantized_model(save_dir)
            if cache_dir is not None:
                shutil.rmtree(cache_dir)
        else:
            float_prog, _ = slim.quant.convert(
                self.test_prog, self.exe.place, save_int8=True)
            test_input_names = [
                var.name for var in list(self.test_inputs.values())
            ]
            test_outputs = list(self.test_outputs.values())
            fluid.io.save_inference_model(
                dirname=save_dir,
                executor=self.exe,
                params_filename='__params__',
                feeded_var_names=test_input_names,
                target_vars=test_outputs,
                main_program=float_prog)

        model_info = self.get_model_info()
        model_info['status'] = 'Quant'

        # Save input and output descrition of model
        model_info['_ModelInputsOutputs'] = dict()
        model_info['_ModelInputsOutputs']['test_inputs'] = [
            [k, v.name] for k, v in self.test_inputs.items()
        ]
        model_info['_ModelInputsOutputs']['test_outputs'] = [
            [k, v.name] for k, v in self.test_outputs.items()
        ]

        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        # The flag of model for saving successfully
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model for quant saved in {}.".format(save_dir))

    def default_optimizer(self,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          lr_decay_power=0.9,
                          regularization_coeff=4e-5):
        decay_step = num_epochs * num_steps_each_epoch
        lr_decay = fluid.layers.polynomial_decay(
            learning_rate,
            decay_step,
            end_learning_rate=0,
            power=lr_decay_power)
        optimizer = fluid.optimizer.Momentum(
            lr_decay,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=regularization_coeff))
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrained_weights=None,
              resume_weights=None,
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              regularization_coeff=4e-5,
              use_vdl=False,
              quant=False):
        self.labels = train_dataset.labels
        self.train_transforms = train_dataset.transforms
        self.train_init = locals()
        self.begin_epoch = 0

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            optimizer = self.default_optimizer(
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                num_steps_each_epoch=num_steps_each_epoch,
                lr_decay_power=lr_decay_power,
                regularization_coeff=regularization_coeff)
        self.optimizer = optimizer
        self.build_program()
        self.net_initialize(
            startup_prog=fluid.default_startup_program(),
            pretrained_weights=pretrained_weights,
            resume_weights=resume_weights)

        # 进行量化
        if quant:
            # 当 for_test=False ，返回类型为 fluid.CompiledProgram
            # 当 for_test=True ，返回类型为 fluid.Program
            self.train_prog = slim.quant.quant_aware(
                self.train_prog, self.exe.place, for_test=False)
            self.test_prog = slim.quant.quant_aware(
                self.test_prog, self.exe.place, for_test=True)
            # self.parallel_train_prog = self.train_prog.with_data_parallel(
            #     loss_name=self.train_outputs['loss'].name)
            self.status = 'Quant'

        if self.begin_epoch >= num_epochs:
            raise ValueError(
                ("begin epoch[{}] is larger than num_epochs[{}]").format(
                    self.begin_epoch, num_epochs))

        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)

        # add arrange op tor transforms
        self.arrange_transform(
            transforms=train_dataset.transforms, mode='train')
        self.build_train_data_loader(
            dataset=train_dataset, batch_size=train_batch_size)

        if eval_dataset is not None:
            self.eval_transforms = eval_dataset.transforms
            self.test_transforms = copy.deepcopy(eval_dataset.transforms)

        lr = self.optimizer._learning_rate
        lr.persistable = True
        if isinstance(lr, fluid.framework.Variable):
            self.train_outputs['lr'] = lr

        # 多卡训练
        if self.parallel_train_prog is None:
            build_strategy = fluid.compiler.BuildStrategy()
            if self.env_info['place'] != 'cpu' and len(self.places) > 1:
                build_strategy.sync_batch_norm = self.sync_bn
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_iteration_per_drop_scope = 1
            if quant:
                build_strategy.fuse_all_reduce_ops = False
                build_strategy.sync_batch_norm = False
                self.parallel_train_prog = self.train_prog.with_data_parallel(
                    loss_name=self.train_outputs['loss'].name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)
            else:
                self.parallel_train_prog = fluid.CompiledProgram(
                    self.train_prog).with_data_parallel(
                        loss_name=self.train_outputs['loss'].name,
                        build_strategy=build_strategy,
                        exec_strategy=exec_strategy)

        total_num_steps = math.floor(
            train_dataset.num_samples / train_batch_size)
        num_steps = 0
        time_stat = list()
        time_train_one_epoch = None
        time_eval_one_epoch = None

        total_num_steps_eval = 0
        # eval times
        total_eval_times = math.ceil(num_epochs / save_interval_epochs)
        eval_batch_size = train_batch_size
        if eval_dataset is not None:
            total_num_steps_eval = math.ceil(
                eval_dataset.num_samples / eval_batch_size)

        if use_vdl:
            from visualdl import LogWriter
            vdl_logdir = osp.join(save_dir, 'vdl_log')
            log_writer = LogWriter(vdl_logdir)
        best_miou = -1.0
        best_model_epoch = 1
        for i in range(self.begin_epoch, num_epochs):
            records = list()
            step_start_time = time.time()
            epoch_start_time = time.time()
            for step, data in enumerate(self.train_data_loader()):
                outputs = self.exe.run(
                    self.parallel_train_prog,
                    feed=data,
                    fetch_list=list(self.train_outputs.values()))
                outputs_avg = np.mean(np.array(outputs), axis=1)
                records.append(outputs_avg)

                # time estimated to complete the training
                currend_time = time.time()
                step_cost_time = currend_time - step_start_time
                step_start_time = currend_time
                if len(time_stat) < 20:
                    time_stat.append(step_cost_time)
                else:
                    time_stat[num_steps % 20] = step_cost_time

                num_steps += 1
                if num_steps % log_interval_steps == 0:
                    step_metrics = OrderedDict(
                        zip(list(self.train_outputs.keys()), outputs_avg))

                    if use_vdl:
                        for k, v in step_metrics.items():
                            log_writer.add_scalar(
                                step=num_steps,
                                tag='train/{}'.format(k),
                                value=v)

                    # 计算剩余时间
                    avg_step_time = np.mean(time_stat)
                    if time_train_one_epoch is not None:
                        eta = (num_epochs - i - 1) * time_train_one_epoch + (
                            total_num_steps - step - 1) * avg_step_time
                    else:
                        eta = ((num_epochs - i) * total_num_steps - step -
                               1) * avg_step_time
                    if time_eval_one_epoch is not None:
                        eval_eta = (total_eval_times - i // save_interval_epochs
                                    ) * time_eval_one_epoch
                    else:
                        eval_eta = (total_eval_times - i // save_interval_epochs
                                    ) * total_num_steps_eval * avg_step_time
                    eta_str = seconds_to_hms(eta + eval_eta)

                    logging.info(
                        "[TRAIN] Epoch={}/{}, Step={}/{}, {}, time_each_step={}s, eta={}"
                        .format(i + 1, num_epochs, step + 1, total_num_steps,
                                dict2str(step_metrics), round(avg_step_time, 2),
                                eta_str))

            train_metrics = OrderedDict(
                zip(list(self.train_outputs.keys()), np.mean(records, axis=0)))
            logging.info('[TRAIN] Epoch {} finished, {} .'.format(
                i + 1, dict2str(train_metrics)))
            time_train_one_epoch = time.time() - epoch_start_time

            eval_epoch_start_time = time.time()
            if (i + 1) % save_interval_epochs == 0 or i == num_epochs - 1:
                current_save_dir = osp.join(save_dir, "epoch_{}".format(i + 1))
                if not osp.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                if eval_dataset is not None:
                    self.eval_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        batch_size=eval_batch_size,
                        epoch_id=i + 1)
                    # 保存最优模型
                    current_miou = self.eval_metrics['miou']
                    if current_miou > best_miou:
                        best_miou = current_miou
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
                            log_writer.add_scalar(
                                step=num_steps,
                                tag='evaluate/{}'.format(k),
                                value=v)
                self.save_model(save_dir=current_save_dir)
                time_eval_one_epoch = time.time() - eval_epoch_start_time
                if eval_dataset is not None:
                    logging.info(
                        'Current evaluated best model in eval_dataset is epoch_{}, miou={}'
                        .format(best_model_epoch, best_miou))

        if quant:
            if osp.exists(osp.join(save_dir, "best_model")):
                fluid.load(
                    program=self.test_prog,
                    model_path=osp.join(save_dir, "best_model"),
                    executor=self.exe)
            self.export_quant_model(
                save_dir=osp.join(save_dir, "best_model_export"),
                quant_type="online")

    def evaluate(self, eval_dataset, batch_size=1, epoch_id=None):
        """评估。

        Args:
            eval_dataset (paddlex.datasets): 评估数据读取器。
            batch_size (int): 评估时的batch大小。默认1。
            epoch_id (int): 当前评估模型所在的训练轮数。
            return_details (bool): 是否返回详细信息。默认False。

        Returns:
            dict: 当return_details为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
                'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
            tuple (metrics, eval_details)：当return_details为True时，增加返回dict (eval_details)，
                包含关键字：'confusion_matrix'，表示评估的混淆矩阵。
        """
        self.arrange_transform(transforms=eval_dataset.transforms, mode='train')
        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        conf_mat = ConfusionMatrix(self.num_classes, streaming=True)
        data_generator = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)
        if not hasattr(self, 'parallel_test_prog'):
            self.parallel_test_prog = fluid.CompiledProgram(
                self.test_prog).with_data_parallel(
                    share_vars_from=self.parallel_train_prog)
        logging.info(
            "Start to evaluating(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(
                enumerate(data_generator()), total=total_steps):
            images = np.array([d[0] for d in data])
            labels = np.array([d[1] for d in data])
            num_samples = images.shape[0]
            if num_samples < batch_size:
                num_pad_samples = batch_size - num_samples
                pad_images = np.tile(images[0:1], (num_pad_samples, 1, 1, 1))
                images = np.concatenate([images, pad_images])
            feed_data = {'image': images}
            outputs = self.exe.run(
                self.parallel_test_prog,
                feed=feed_data,
                fetch_list=list(self.test_outputs.values()),
                return_numpy=True)
            pred = outputs[0]
            if num_samples < batch_size:
                pred = pred[0:num_samples]

            mask = labels != self.ignore_index
            conf_mat.calculate(pred=pred, label=labels, ignore=mask)
            _, iou = conf_mat.mean_iou()

            logging.debug("[EVAL] Epoch={}, Step={}/{}, iou={}".format(
                epoch_id, step + 1, total_steps, iou))

        category_iou, miou = conf_mat.mean_iou()
        category_acc, macc = conf_mat.accuracy()

        metrics = OrderedDict(
            zip(['miou', 'category_iou', 'macc', 'category_acc', 'kappa'],
                [miou, category_iou, macc, category_acc,
                 conf_mat.kappa()]))

        logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
            epoch_id, dict2str(metrics)))
        return metrics

    def predict(self, im_file, transforms=None):
        """预测。
        Args:
            img_file(str|np.ndarray): 预测图像。
            transforms(paddlex.cv.transforms): 数据预处理操作。

        Returns:
            dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """
        if isinstance(im_file, str):
            if not osp.exists(im_file):
                raise ValueError(
                    'The Image file does not exist: {}'.format(im_file))

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is not None:
            self.arrange_transform(transforms=transforms, mode='test')
            im, im_info = transforms(im_file)
        else:
            self.arrange_transform(transforms=self.test_transforms, mode='test')
            im, im_info = self.test_transforms(im_file)
        im = np.expand_dims(im, axis=0)
        result = self.exe.run(
            self.test_prog,
            feed={'image': im},
            fetch_list=list(self.test_outputs.values()))
        pred = result[0]
        logit = result[1]
        logit = np.squeeze(logit)
        logit = np.transpose(logit, (1, 2, 0))
        pred = np.squeeze(pred).astype('uint8')
        keys = list(im_info.keys())
        for k in keys[::-1]:
            if k == 'shape_before_resize':
                h, w = im_info[k][0], im_info[k][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
                logit = cv2.resize(logit, (w, h), cv2.INTER_LINEAR)
            elif k == 'shape_before_padding':
                h, w = im_info[k][0], im_info[k][1]
                pred = pred[0:h, 0:w]
                logit = logit[0:h, 0:w, :]

        return {'label_map': pred, 'score_map': logit}


class HumanSegLite(SegModel):
    # DeepLab ShuffleNet
    def build_net(self, mode='train'):
        """应根据不同的情况进行构建"""
        model = ShuffleSeg(
            self.num_classes,
            mode=mode,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs


class HumanSegServer(SegModel):
    # DeepLab Xception
    def __init__(self,
                 num_classes=2,
                 backbone='Xception65',
                 output_stride=16,
                 aspp_with_sep_conv=True,
                 decoder_use_sep_conv=True,
                 encoder_with_aspp=True,
                 enable_decoder=True,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 sync_bn=True):
        super().__init__(
            num_classes=num_classes,
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss,
            class_weight=class_weight,
            ignore_index=ignore_index,
            sync_bn=sync_bn)
        self.init_params = locals()

        self.output_stride = output_stride

        if backbone not in ['Xception65', 'Xception41']:
            raise ValueError("backbone: {} is set wrong. it should be one of "
                             "('Xception65', 'Xception41')".format(backbone))

        self.backbone = backbone
        self.aspp_with_sep_conv = aspp_with_sep_conv
        self.decoder_use_sep_conv = decoder_use_sep_conv
        self.encoder_with_aspp = encoder_with_aspp
        self.enable_decoder = enable_decoder
        self.sync_bn = sync_bn

    def build_net(self, mode='train'):
        model = DeepLabv3p(
            self.num_classes,
            mode=mode,
            backbone=self.backbone,
            output_stride=self.output_stride,
            aspp_with_sep_conv=self.aspp_with_sep_conv,
            decoder_use_sep_conv=self.decoder_use_sep_conv,
            encoder_with_aspp=self.encoder_with_aspp,
            enable_decoder=self.enable_decoder,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs


class HumanSegMobile(SegModel):
    def __init__(self,
                 num_classes=2,
                 stage1_num_modules=1,
                 stage1_num_blocks=[1],
                 stage1_num_channels=[32],
                 stage2_num_modules=1,
                 stage2_num_blocks=[2, 2],
                 stage2_num_channels=[16, 32],
                 stage3_num_modules=1,
                 stage3_num_blocks=[2, 2, 2],
                 stage3_num_channels=[16, 32, 64],
                 stage4_num_modules=1,
                 stage4_num_blocks=[2, 2, 2, 2],
                 stage4_num_channels=[16, 32, 64, 128],
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 sync_bn=True):
        super().__init__(
            num_classes=num_classes,
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss,
            class_weight=class_weight,
            ignore_index=ignore_index,
            sync_bn=sync_bn)
        self.init_params = locals()

        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels

    def build_net(self, mode='train'):
        """应根据不同的情况进行构建"""
        model = HRNet(
            self.num_classes,
            mode=mode,
            stage1_num_modules=self.stage1_num_modules,
            stage1_num_blocks=self.stage1_num_blocks,
            stage1_num_channels=self.stage1_num_channels,
            stage2_num_modules=self.stage2_num_modules,
            stage2_num_blocks=self.stage2_num_blocks,
            stage2_num_channels=self.stage2_num_channels,
            stage3_num_modules=self.stage3_num_modules,
            stage3_num_blocks=self.stage3_num_blocks,
            stage3_num_channels=self.stage3_num_channels,
            stage4_num_modules=self.stage4_num_modules,
            stage4_num_blocks=self.stage4_num_blocks,
            stage4_num_channels=self.stage4_num_channels,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrained_weights=None,
              resume_weights=None,
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              regularization_coeff=5e-4,
              use_vdl=False,
              quant=False):
        super().train(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrained_weights=pretrained_weights,
            resume_weights=resume_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            regularization_coeff=regularization_coeff,
            use_vdl=use_vdl,
            quant=quant)
