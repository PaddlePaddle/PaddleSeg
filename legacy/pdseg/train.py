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
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import sys
import argparse
import pprint
import random
import shutil
import time

import numpy as np
import paddle
import paddle.static as static
from paddle.fluid import profiler
import paddle.distributed.fleet as fleet

from utils.config import cfg
from utils.timer import TimeAverager, calculate_eta
from metrics import ConfusionMatrix
from reader import SegDataset
from models.model_builder import build_model
from models.model_builder import ModelPhase
from eval import evaluate
from vis import visualize
from utils import dist_utils
from utils.load_model_utils import load_pretrained_weights
from utils import paddle_utils


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
        help='Use gpu, xpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_xpu',
        dest='use_xpu',
        help='Use xpu, gpu or cpu',
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
        '--use_vdl',
        dest='use_vdl',
        help='whether to record the data during training to VisualDL',
        action='store_true')
    parser.add_argument(
        '--vdl_log_dir',
        dest='vdl_log_dir',
        help='VisualDL logging directory',
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

    # NOTE: This for benchmark
    parser.add_argument(
        '--is_profiler',
        help='the profiler switch.(used for benchmark)',
        default=0,
        type=int)
    parser.add_argument(
        '--profiler_path',
        help='the profiler output file path.(used for benchmark)',
        default='./seg.profiler',
        type=str)
    return parser.parse_args()


def save_checkpoint(program, ckpt_name):
    """
    Save checkpoint for evaluation or resume training
    """
    ckpt_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, str(ckpt_name))
    print("Save model checkpoint to {}".format(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    static.save(program, os.path.join(ckpt_dir, 'model'))

    return ckpt_dir


def load_checkpoint(exe, program):
    """
    Load checkpoiont for resuming training
    """
    model_path = cfg.TRAIN.RESUME_MODEL_DIR
    print('Resume model training from:', model_path)
    if not os.path.exists(model_path):
        raise ValueError(
            "TRAIN.PRETRAIN_MODEL {} not exist!".format(model_path))
    static.load(program, os.path.join(model_path, 'model'), exe)

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


def save_infer_program(test_program, ckpt_dir):
    _test_program = test_program.clone()
    _test_program.desc.flush()
    _test_program.desc._set_version()
    paddle_utils.save_op_version_info(_test_program.desc)
    with open(os.path.join(ckpt_dir, 'model') + ".pdmodel", "wb") as f:
        f.write(_test_program.desc.serialize_to_string())


def update_best_model(ckpt_dir):
    best_model_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'best_model')
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    shutil.copytree(ckpt_dir, best_model_dir)


def print_info(*msg):
    if cfg.TRAINER_ID == 0:
        print(*msg)


def train(cfg):
    # Use the default program for fleetrun
    startup_prog = static.default_startup_program()
    train_prog = static.default_main_program()
    test_prog = static.Program()
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
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    xpu_id = int(os.environ.get('FLAGS_selected_xpus', 0))
    if args.use_gpu:
        place = paddle.CUDAPlace(gpu_id)
        places = static.cuda_places()
    elif args.use_xpu:
        place = paddle.XPUPlace(xpu_id)
        places = [place]
    else:
        place = paddle.CPUPlace()
        places = static.cpu_places()

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

    data_loader, avg_loss, lr, pred, grts, masks, optimizer, _new_generator = build_model(
        train_prog, startup_prog, phase=ModelPhase.TRAIN)
    build_model(test_prog, static.Program(), phase=ModelPhase.EVAL)
    data_loader.set_sample_generator(
        data_generator, batch_size=batch_size_per_dev, drop_last=drop_last)

    exec_strategy = static.ExecutionStrategy()
    # Clear temporary variables every 100 iteration
    if args.use_gpu:
        exec_strategy.num_threads = len(paddle.get_cuda_rng_state())
    exec_strategy.num_iteration_per_drop_scope = 100
    build_strategy = static.BuildStrategy()

    if cfg.TRAIN.SYNC_BATCH_NORM and args.use_gpu:
        if dev_count > 1:
            # Apply sync batch norm strategy
            print_info("Sync BatchNorm strategy is effective.")
            build_strategy.sync_batch_norm = True
        else:
            print_info(
                "Sync BatchNorm strategy will not be effective if GPU device"
                " count <= 1")

    if cfg.NUM_TRAINERS > 1 and args.use_gpu:
        strategy = fleet.DistributedStrategy()
        strategy.sync_batch_norm = True
        exec_strategy.num_threads = 1
        strategy.execution_strategy = exec_strategy
        strategy.build_strategy = build_strategy
        strategy.cudnn_exhaustive_search = False
        strategy.cudnn_batchnorm_spatial_persistent = False
        strategy.conv_workspace_size_limit = 512
        fleet.init(is_collective=True, strategy=strategy)
        optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)

    with paddle.utils.unique_name.guard(_new_generator):
        optimizer.minimize(avg_loss)
    # if cfg.NUM_TRAINERS > 1 and args.use_gpu:
    #     dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
    #     exec_strategy.num_threads = 1

    with open("train_prog_{}".format(cfg.NUM_TRAINERS), "w") as f:
        if cfg.TRAINER_ID == 0:
            f.writelines(str(train_prog))

    if args.use_xpu or (cfg.NUM_TRAINERS > 1 and args.use_gpu):
        compiled_train_prog = train_prog
    else:
        compiled_train_prog = static.CompiledProgram(
            train_prog).with_data_parallel(
                loss_name=avg_loss.name,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)

    exe = static.Executor(place)
    exe.run(startup_prog)

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

    fetch_list = [avg_loss.name]

    if args.use_vdl:
        if not args.vdl_log_dir:
            print_info("Please specify the log directory by --vdl_log_dir.")
            exit(1)

        from visualdl import LogWriter
        log_writer = LogWriter(args.vdl_log_dir)

    # trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    # num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    step = 0
    all_step = cfg.DATASET.TRAIN_TOTAL_IMAGES // cfg.BATCH_SIZE
    if cfg.DATASET.TRAIN_TOTAL_IMAGES % cfg.BATCH_SIZE and drop_last != True:
        all_step += 1
    all_step *= (cfg.SOLVER.NUM_EPOCHS - begin_epoch + 1)

    avg_loss = 0.0
    best_mIoU = 0.0
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()

    batch_start = time.time()
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
                reader_cost_averager.record(time.time() - batch_start)
                loss = exe.run(
                    program=compiled_train_prog,
                    fetch_list=fetch_list,
                    return_numpy=True)
                avg_loss += np.mean(np.array(loss))
                step += 1
                batch_cost_averager.record(
                    time.time() - batch_start,
                    num_samples=cfg.BATCH_SIZE / dev_count)

                if step % args.log_steps == 0 and cfg.TRAINER_ID == 0:
                    avg_train_batch_cost = batch_cost_averager.get_average()
                    avg_train_reader_cost = reader_cost_averager.get_average()
                    eta = calculate_eta(all_step - step, avg_train_batch_cost)
                    avg_loss /= args.log_steps
                    print(
                        "epoch: {} step: {} lr: {:.5f} loss: {:.4f} batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                        .format(epoch, step, lr.get_lr(), avg_loss,
                                avg_train_batch_cost, avg_train_reader_cost,
                                batch_cost_averager.get_ips_average(), eta))
                    if args.use_vdl:
                        log_writer.add_scalar('Train/loss', avg_loss, step)
                        log_writer.add_scalar('Train/lr', lr.get_lr(), step)
                        log_writer.add_scalar('Train/batch_cost',
                                              avg_train_batch_cost, step)
                        log_writer.add_scalar('Train/reader_cost',
                                              avg_train_reader_cost, step)
                    sys.stdout.flush()
                    avg_loss = 0.0
                    reader_cost_averager.reset()
                    batch_cost_averager.reset()
                batch_start = time.time()

                # NOTE : used for benchmark, profiler tools
                if args.is_profiler and epoch == 1 and step == args.log_steps:
                    profiler.start_profiler("All")
                elif args.is_profiler and epoch == 1 and step == args.log_steps + 5:
                    profiler.stop_profiler("total", args.profiler_path)
                    return
                lr.step()

            except paddle.fluid.core.EOFException:
                data_loader.reset()
                break
            except Exception as e:
                print(e)

        if (epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0
                or epoch == cfg.SOLVER.NUM_EPOCHS) and cfg.TRAINER_ID == 0:
            ckpt_dir = save_checkpoint(train_prog, epoch)
            save_infer_program(test_prog, ckpt_dir)

            if args.do_eval:
                tmp = cfg.BATCH_SIZE
                cfg.BATCH_SIZE = batch_size_per_dev
                print("Evaluation start")
                _, mean_iou, _, mean_acc = evaluate(
                    cfg=cfg,
                    ckpt_dir=ckpt_dir,
                    use_gpu=args.use_gpu,
                    use_xpu=args.use_xpu,
                    use_mpio=args.use_mpio)
                if args.use_vdl:
                    log_writer.add_scalar('Evaluate/mean_iou', mean_iou, step)
                    log_writer.add_scalar('Evaluate/mean_acc', mean_acc, step)

                if mean_iou > best_mIoU:
                    best_mIoU = mean_iou
                    update_best_model(ckpt_dir)
                    print_info("Save best model {} to {}, mIoU = {:.4f}".format(
                        ckpt_dir,
                        os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'best_model'),
                        mean_iou))
                cfg.BATCH_SIZE = tmp

            # Use VisualDL to visualize results
            if args.use_vdl and cfg.DATASET.VIS_FILE_LIST is not None:
                visualize(
                    cfg=cfg,
                    use_gpu=args.use_gpu,
                    vis_file_list=cfg.DATASET.VIS_FILE_LIST,
                    vis_dir="visual",
                    ckpt_dir=ckpt_dir,
                    log_writer=log_writer)

    # save final model
    if cfg.TRAINER_ID == 0:
        ckpt_dir = save_checkpoint(train_prog, 'final')
        save_infer_program(test_prog, ckpt_dir)


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
    print('************NUM_TRAINERS**********', cfg.NUM_TRAINERS)

    cfg.check_and_infer()
    print_info(pprint.pformat(cfg))
    train(cfg)


if __name__ == '__main__':
    paddle_utils.enable_static()
    args = parse_args()
    if paddle.is_compiled_with_cuda() != True and args.use_gpu == True:
        print(
            "You can not set use_gpu = True in the model because you are using paddlepaddle-cpu."
        )
        print(
            "Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_gpu=False to run models on CPU."
        )
        sys.exit(1)
    main(args)
