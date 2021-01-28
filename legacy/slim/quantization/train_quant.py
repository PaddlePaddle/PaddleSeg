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
from __future__ import division
from __future__ import print_function

import os

import sys
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
from models.model_builder import build_model
from models.model_builder import ModelPhase
from models.model_builder import parse_shape_from_file
from eval_quant import evaluate
from vis import visualize
from utils import dist_utils
from utils.load_model_utils import load_pretrained_weights
from utils import paddle_utils
from train import update_best_model, print_info

from paddleslim.quant import quant_aware


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
    parser.add_argument(
        "--not_quant_pattern",
        nargs='+',
        type=str,
        help=
        "Layers which name_scope contains string in not_quant_pattern will not be quantized"
    )

    return parser.parse_args()


def save_checkpoint(exe, program, ckpt_name):
    """
    Save checkpoint for evaluation or resume training
    """
    ckpt_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, str(ckpt_name))
    print("Save model checkpoint to {}".format(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    fluid.io.save_vars(
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


def train_quant(cfg):
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

    data_loader, avg_loss, lr, pred, grts, masks = build_model(
        train_prog, startup_prog, phase=ModelPhase.TRAIN)
    data_loader.set_sample_generator(
        data_generator, batch_size=batch_size_per_dev, drop_last=drop_last)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    exec_strategy = fluid.ExecutionStrategy()
    # Clear temporary variables every 100 iteration
    if args.use_gpu:
        exec_strategy.num_threads = fluid.core.get_cuda_device_count()
    exec_strategy.num_iteration_per_drop_scope = 100
    build_strategy = fluid.BuildStrategy()

    if cfg.NUM_TRAINERS > 1 and args.use_gpu:
        dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
        exec_strategy.num_threads = 1

    # Resume training
    begin_epoch = cfg.SOLVER.BEGIN_EPOCH
    if cfg.TRAIN.RESUME_MODEL_DIR:
        begin_epoch = load_checkpoint(exe, train_prog)
    # Load pretrained model
    elif os.path.exists(cfg.TRAIN.PRETRAINED_MODEL_DIR):
        load_pretrained_weights(exe, train_prog, cfg.TRAIN.PRETRAINED_MODEL_DIR)
    else:
        print_info(
            'Pretrained model dir {} not exists, training from scratch...'.
            format(cfg.TRAIN.PRETRAINED_MODEL_DIR))

    fetch_list = [avg_loss.name, lr.name]
    if args.debug:
        # Fetch more variable info and use streaming confusion matrix to
        # calculate IoU results if in debug mode
        np.set_printoptions(
            precision=4, suppress=True, linewidth=160, floatmode="fixed")
        fetch_list.extend([pred.name, grts.name, masks.name])
        cm = ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=True)

    not_quant_pattern = []
    if args.not_quant_pattern:
        not_quant_pattern = args.not_quant_pattern
    config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        'not_quant_pattern': not_quant_pattern
    }
    compiled_train_prog = quant_aware(train_prog, place, config, for_test=False)
    eval_prog = quant_aware(train_prog, place, config, for_test=True)
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    compiled_train_prog = compiled_train_prog.with_data_parallel(
        loss_name=avg_loss.name,
        exec_strategy=exec_strategy,
        build_strategy=build_strategy)

    # trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    # num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    global_step = 0
    all_step = cfg.DATASET.TRAIN_TOTAL_IMAGES // cfg.BATCH_SIZE
    if cfg.DATASET.TRAIN_TOTAL_IMAGES % cfg.BATCH_SIZE and drop_last != True:
        all_step += 1
    all_step *= (cfg.SOLVER.NUM_EPOCHS - begin_epoch + 1)

    avg_loss = 0.0
    best_mIoU = 0.0

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

    for epoch in range(begin_epoch, cfg.SOLVER.NUM_EPOCHS + 1):
        data_loader.start()
        while True:
            try:
                if args.debug:
                    # Print category IoU and accuracy to check whether the
                    # traning process is corresponed to expectation
                    loss, lr, pred, grts, masks = exe.run(
                        program=compiled_train_prog,
                        fetch_list=fetch_list,
                        return_numpy=True)
                    cm.calculate(pred, grts, masks)
                    avg_loss += np.mean(np.array(loss))
                    global_step += 1

                    if global_step % args.log_steps == 0:
                        speed = args.log_steps / timer.elapsed_time()
                        avg_loss /= args.log_steps
                        category_acc, mean_acc = cm.accuracy()
                        category_iou, mean_iou = cm.mean_iou()

                        print_info((
                            "epoch={} step={} lr={:.5f} loss={:.4f} acc={:.5f} mIoU={:.5f} step/sec={:.3f} | ETA {}"
                        ).format(epoch, global_step, lr[0], avg_loss, mean_acc,
                                 mean_iou, speed,
                                 calculate_eta(all_step - global_step, speed)))
                        print_info("Category IoU: ", category_iou)
                        print_info("Category Acc: ", category_acc)
                        sys.stdout.flush()
                        avg_loss = 0.0
                        cm.zero_matrix()
                        timer.restart()
                else:
                    # If not in debug mode, avoid unnessary log and calculate
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
                data_loader.reset()
                break
            except Exception as e:
                print(e)

        if (epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0
                or epoch == cfg.SOLVER.NUM_EPOCHS) and cfg.TRAINER_ID == 0:
            ckpt_dir = save_checkpoint(exe, eval_prog, epoch)

            if args.do_eval:
                print("Evaluation start")
                _, mean_iou, _, mean_acc = evaluate(
                    cfg=cfg,
                    ckpt_dir=ckpt_dir,
                    use_gpu=args.use_gpu,
                    use_mpio=args.use_mpio,
                    not_quant_pattern=args.not_quant_pattern,
                    convert=False)

                if mean_iou > best_mIoU:
                    best_mIoU = mean_iou
                    update_best_model(ckpt_dir)
                    print_info("Save best model {} to {}, mIoU = {:.4f}".format(
                        ckpt_dir,
                        os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'best_model'),
                        mean_iou))

    # save final model
    if cfg.TRAINER_ID == 0:
        save_checkpoint(exe, eval_prog, 'final')


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
    train_quant(cfg)


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
    paddle_utils.enable_static()
    main(args)
