# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import division
from __future__ import print_function

import os
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
SEG_PATH = os.path.join(LOCAL_PATH, "../../", "pdseg")
sys.path.append(SEG_PATH)

import argparse
import pprint
import random
import shutil
import functools

import paddle
import numpy as np
import paddle.fluid as fluid

from utils.config import cfg
from utils.timer import Timer, calculate_eta
from metrics import ConfusionMatrix
from reader import SegDataset
from model_builder import build_model
from model_builder import ModelPhase
from model_builder import parse_shape_from_file
from eval_nas import evaluate
from vis import visualize
from utils import dist_utils

from mobilenetv2_search_space import MobileNetV2SpaceSeg
from paddleslim.nas.search_space.search_space_factory import SearchSpaceFactory
from paddleslim.analysis import flops
from paddleslim.nas.sa_nas import SANAS
from paddleslim.nas import search_space

def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg training')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess I/O or not',
        action='store_true',
        default=False)
    parser.add_argument(
        '--log_steps',
        dest='log_steps',
        help='Display logging information at every log_steps',
        default=10,
        type=int)
    parser.add_argument(
        '--debug',
        dest='debug',
        help='debug mode, display detail information of training',
        action='store_true')
    parser.add_argument(
        '--use_tb',
        dest='use_tb',
        help='whether to record the data during training to Tensorboard',
        action='store_true')
    parser.add_argument(
        '--tb_log_dir',
        dest='tb_log_dir',
        help='Tensorboard logging directory',
        default=None,
        type=str)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Evaluation models result on every new checkpoint',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--enable_ce',
        dest='enable_ce',
        help='If set True, enable continuous evaluation job.'
        'This flag is only used for internal test.',
        action='store_true')
    return parser.parse_args()


def save_vars(executor, dirname, program=None, vars=None):
    """
    Temporary resolution for Win save variables compatability.
    Will fix in PaddlePaddle v1.5.2
    """

    save_program = fluid.Program()
    save_block = save_program.global_block()

    for each_var in vars:
        # NOTE: don't save the variable which type is RAW
        if each_var.type == fluid.core.VarDesc.VarType.RAW:
            continue
        new_var = save_block.create_var(
            name=each_var.name,
            shape=each_var.shape,
            dtype=each_var.dtype,
            type=each_var.type,
            lod_level=each_var.lod_level,
            persistable=True)
        file_path = os.path.join(dirname, new_var.name)
        file_path = os.path.normpath(file_path)
        save_block.append_op(
            type='save',
            inputs={'X': [new_var]},
            outputs={},
            attrs={'file_path': file_path})

    executor.run(save_program)


def save_checkpoint(exe, program, ckpt_name):
    """
    Save checkpoint for evaluation or resume training
    """
    ckpt_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, str(ckpt_name))
    print("Save model checkpoint to {}".format(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    save_vars(
        exe,
        ckpt_dir,
        program,
        vars=list(filter(fluid.io.is_persistable, program.list_vars())))

    return ckpt_dir


def load_checkpoint(exe, program):
    """
    Load checkpoiont from pretrained model directory for resume training
    """

    print('Resume model training from:', cfg.TRAIN.RESUME_MODEL_DIR)
    if not os.path.exists(cfg.TRAIN.RESUME_MODEL_DIR):
        raise ValueError("TRAIN.PRETRAIN_MODEL {} not exist!".format(
            cfg.TRAIN.RESUME_MODEL_DIR))

    fluid.io.load_persistables(
        exe, cfg.TRAIN.RESUME_MODEL_DIR, main_program=program)

    model_path = cfg.TRAIN.RESUME_MODEL_DIR
    # Check is path ended by path spearator
    if model_path[-1] == os.sep:
        model_path = model_path[0:-1]
    epoch_name = os.path.basename(model_path)
    # If resume model is final model
    if epoch_name == 'final':
        begin_epoch = cfg.SOLVER.NUM_EPOCHS
    # If resume model path is end of digit, restore epoch status
    elif epoch_name.isdigit():
        epoch = int(epoch_name)
        begin_epoch = epoch + 1
    else:
        raise ValueError("Resume model path is not valid!")
    print("Model checkpoint loaded successfully!")

    return begin_epoch


def update_best_model(ckpt_dir):
    best_model_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'best_model')
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    shutil.copytree(ckpt_dir, best_model_dir)


def print_info(*msg):
    if cfg.TRAINER_ID == 0:
        print(*msg)


def train(cfg):
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000
    drop_last = True

    dataset = SegDataset(
        file_list=cfg.DATASET.TRAIN_FILE_LIST,
        mode=ModelPhase.TRAIN,
        shuffle=True,
        data_dir=cfg.DATASET.DATA_DIR)

    def data_generator():
        if args.use_mpio:
            data_gen = dataset.multiprocess_generator(
                num_processes=cfg.DATALOADER.NUM_WORKERS,
                max_queue_size=cfg.DATALOADER.BUF_SIZE)
        else:
            data_gen = dataset.generator()

        batch_data = []
        for b in data_gen:
            batch_data.append(b)
            if len(batch_data) == (cfg.BATCH_SIZE // cfg.NUM_TRAINERS):
                for item in batch_data:
                    yield item[0], item[1], item[2]
                batch_data = []
        # If use sync batch norm strategy, drop last batch if number of samples
        # in batch_data is less then cfg.BATCH_SIZE to avoid NCCL hang issues
        if not cfg.TRAIN.SYNC_BATCH_NORM:
            for item in batch_data:
                yield item[0], item[1], item[2]

    # Get device environment
    # places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    # place = places[0]
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()

    # Get number of GPU
    dev_count = cfg.NUM_TRAINERS if cfg.NUM_TRAINERS > 1 else len(places)
    print_info("#Device count: {}".format(dev_count))

    # Make sure BATCH_SIZE can divided by GPU cards
    assert cfg.BATCH_SIZE % dev_count == 0, (
        'BATCH_SIZE:{} not divisble by number of GPUs:{}'.format(
            cfg.BATCH_SIZE, dev_count))
    # If use multi-gpu training mode, batch data will allocated to each GPU evenly
    batch_size_per_dev = cfg.BATCH_SIZE // dev_count
    print_info("batch_size_per_dev: {}".format(batch_size_per_dev))

    config_info = {'input_size': 769, 'output_size': 1, 'block_num': 7}
    config = ([(cfg.SLIM.NAS_SPACE_NAME, config_info)])
    factory = SearchSpaceFactory()
    space = factory.get_search_space(config)

    port = cfg.SLIM.NAS_PORT
    server_address = (cfg.SLIM.NAS_ADDRESS, port)
    sa_nas = SANAS(config, server_addr=server_address, search_steps=cfg.SLIM.NAS_SEARCH_STEPS,
                   is_server=cfg.SLIM.NAS_IS_SERVER)
    for step in range(cfg.SLIM.NAS_SEARCH_STEPS):
        arch = sa_nas.next_archs()[0]

        start_prog = fluid.Program()
        train_prog = fluid.Program()

        py_reader, avg_loss, lr, pred, grts, masks = build_model(
            train_prog, start_prog, arch=arch, phase=ModelPhase.TRAIN)

        cur_flops = flops(train_prog)
        print('current step:', step, 'flops:', cur_flops)

        py_reader.decorate_sample_generator(
            data_generator, batch_size=batch_size_per_dev, drop_last=drop_last)

        exe = fluid.Executor(place)
        exe.run(start_prog)

        exec_strategy = fluid.ExecutionStrategy()
        # Clear temporary variables every 100 iteration
        if args.use_gpu:
            exec_strategy.num_threads = fluid.core.get_cuda_device_count()
        exec_strategy.num_iteration_per_drop_scope = 100
        build_strategy = fluid.BuildStrategy()

        if cfg.NUM_TRAINERS > 1 and args.use_gpu:
            dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
            exec_strategy.num_threads = 1

        if cfg.TRAIN.SYNC_BATCH_NORM and args.use_gpu:
            if dev_count > 1:
                # Apply sync batch norm strategy
                print_info("Sync BatchNorm strategy is effective.")
                build_strategy.sync_batch_norm = True
            else:
                print_info(
                    "Sync BatchNorm strategy will not be effective if GPU device"
                    " count <= 1")
        compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
            loss_name=avg_loss.name,
            exec_strategy=exec_strategy,
            build_strategy=build_strategy)

        # Resume training
        begin_epoch = cfg.SOLVER.BEGIN_EPOCH
        if cfg.TRAIN.RESUME_MODEL_DIR:
            begin_epoch = load_checkpoint(exe, train_prog)
        # Load pretrained model
        elif os.path.exists(cfg.TRAIN.PRETRAINED_MODEL_DIR):
            print_info('Pretrained model dir: ', cfg.TRAIN.PRETRAINED_MODEL_DIR)
            load_vars = []
            load_fail_vars = []

            def var_shape_matched(var, shape):
                """
                Check whehter persitable variable shape is match with current network
                """
                var_exist = os.path.exists(
                    os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, var.name))
                if var_exist:
                    var_shape = parse_shape_from_file(
                        os.path.join(cfg.TRAIN.PRETRAINED_MODEL_DIR, var.name))
                    return var_shape == shape
                return False

            for x in train_prog.list_vars():
                if isinstance(x, fluid.framework.Parameter):
                    shape = tuple(fluid.global_scope().find_var(
                        x.name).get_tensor().shape())
                    if var_shape_matched(x, shape):
                        load_vars.append(x)
                    else:
                        load_fail_vars.append(x)

            fluid.io.load_vars(
                exe, dirname=cfg.TRAIN.PRETRAINED_MODEL_DIR, vars=load_vars)
            for var in load_vars:
                print_info("Parameter[{}] loaded sucessfully!".format(var.name))
            for var in load_fail_vars:
                print_info(
                    "Parameter[{}] don't exist or shape does not match current network, skip"
                    " to load it.".format(var.name))
            print_info("{}/{} pretrained parameters loaded successfully!".format(
                len(load_vars),
                len(load_vars) + len(load_fail_vars)))
        else:
            print_info(
                'Pretrained model dir {} not exists, training from scratch...'.
                    format(cfg.TRAIN.PRETRAINED_MODEL_DIR))

        fetch_list = [avg_loss.name, lr.name]

        global_step = 0
        all_step = cfg.DATASET.TRAIN_TOTAL_IMAGES // cfg.BATCH_SIZE
        if cfg.DATASET.TRAIN_TOTAL_IMAGES % cfg.BATCH_SIZE and drop_last != True:
            all_step += 1
        all_step *= (cfg.SOLVER.NUM_EPOCHS - begin_epoch + 1)

        avg_loss = 0.0
        timer = Timer()
        timer.start()
        if begin_epoch > cfg.SOLVER.NUM_EPOCHS:
            raise ValueError(
                ("begin epoch[{}] is larger than cfg.SOLVER.NUM_EPOCHS[{}]").format(
                    begin_epoch, cfg.SOLVER.NUM_EPOCHS))

        if args.use_mpio:
            print_info("Use multiprocess reader")
        else:
            print_info("Use multi-thread reader")

        best_miou = 0.0
        for epoch in range(begin_epoch, cfg.SOLVER.NUM_EPOCHS + 1):
            py_reader.start()
            while True:
                try:
                    loss, lr = exe.run(
                        program=compiled_train_prog,
                        fetch_list=fetch_list,
                        return_numpy=True)
                    avg_loss += np.mean(np.array(loss))
                    global_step += 1

                    if global_step % args.log_steps == 0 and cfg.TRAINER_ID == 0:
                        avg_loss /= args.log_steps
                        speed = args.log_steps / timer.elapsed_time()
                        print((
                                  "epoch={} step={} lr={:.5f} loss={:.4f} step/sec={:.3f} | ETA {}"
                              ).format(epoch, global_step, lr[0], avg_loss, speed,
                                       calculate_eta(all_step - global_step, speed)))

                        sys.stdout.flush()
                        avg_loss = 0.0
                        timer.restart()

                except fluid.core.EOFException:
                    py_reader.reset()
                    break
                except Exception as e:
                    print(e)
            if epoch > cfg.SLIM.NAS_START_EVAL_EPOCH:
                ckpt_dir = save_checkpoint(exe, train_prog, '{}_tmp'.format(port))
                _, mean_iou, _, mean_acc = evaluate(
                    cfg=cfg,
                    arch=arch,
                    ckpt_dir=ckpt_dir,
                    use_gpu=args.use_gpu,
                    use_mpio=args.use_mpio)
                if best_miou < mean_iou:
                    print('search step {}, epoch {} best iou {}'.format(step, epoch, mean_iou))
                    best_miou = mean_iou

        sa_nas.reward(float(best_miou))


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    if args.enable_ce:
        random.seed(0)
        np.random.seed(0)

    cfg.TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", 0))
    cfg.NUM_TRAINERS = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

    cfg.check_and_infer()
    print_info(pprint.pformat(cfg))
    train(cfg)


if __name__ == '__main__':
    args = parse_args()
    if fluid.core.is_compiled_with_cuda() != True and args.use_gpu == True:
        print(
            "You can not set use_gpu = True in the model because you are using paddlepaddle-cpu."
        )
        print(
            "Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_gpu=False to run models on CPU."
        )
        sys.exit(1)
    main(args)
