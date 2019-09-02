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
import argparse
import pprint
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
from eval import evaluate
from vis import visualize
from utils.fp16_utils import load_fp16_vars


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

    print('Resume model training from:', cfg.TRAIN.PRETRAINED_MODEL)
    if not os.path.exists(cfg.TRAIN.PRETRAINED_MODEL):
        raise ValueError("TRAIN.PRETRAIN_MODEL {} not exist!".format(
            cfg.TRAIN.PRETRAINED_MODEL))

    fluid.io.load_persistables(
        exe, cfg.TRAIN.PRETRAINED_MODEL, main_program=program)

    model_path = cfg.TRAIN.PRETRAINED_MODEL
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


def train(cfg):
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
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
            if len(batch_data) == cfg.BATCH_SIZE:
                for item in batch_data:
                    yield item[0], item[1], item[2]
                batch_data = []
        # If use sync batch norm strategy, drop last batch if number of samples
        # in batch_data is less then cfg.BATCH_SIZE to avoid NCCL hang issues
        if not cfg.TRAIN.SYNC_BATCH_NORM:
            for item in batch_data:
                yield item[0], item[1], item[2]

    # Get device environment
    places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    place = places[0]
    # Get number of GPU
    dev_count = len(places)
    print("#GPU-Devices: {}".format(dev_count))

    # Make sure BATCH_SIZE can divided by GPU cards
    assert cfg.BATCH_SIZE % dev_count == 0, (
        'BATCH_SIZE:{} not divisble by number of GPUs:{}'.format(
            cfg.BATCH_SIZE, dev_count))
    # If use multi-gpu training mode, batch data will allocated to each GPU evenly
    batch_size_per_dev = cfg.BATCH_SIZE // dev_count
    print("batch_size_per_dev: {}".format(batch_size_per_dev))

    document_type = '_'.join([cfg.SOLVER.OPTIMIZER,cfg.SOLVER.LR_POLICY, \
            "batch_size", str(cfg.BATCH_SIZE), "GPUs", str(dev_count),\
            "epochs",str(cfg.SOLVER.NUM_EPOCHS), "lr", str(cfg.SOLVER.LR)])
    cfg.TRAIN.MODEL_SAVE_DIR = os.path.join( cfg.TRAIN.MODEL_SAVE_DIR, document_type)

    print(cfg.TRAIN.MODEL_SAVE_DIR)
    py_reader, avg_loss, lr, pred, grts, masks = build_model(
        train_prog, startup_prog, phase=ModelPhase.TRAIN)
    py_reader.decorate_sample_generator(
        data_generator, batch_size=batch_size_per_dev, drop_last=drop_last)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    exec_strategy = fluid.ExecutionStrategy()
    # Clear temporary variables every 100 iteration
    if args.use_gpu:
        exec_strategy.num_threads = fluid.core.get_cuda_device_count()
    exec_strategy.num_iteration_per_drop_scope = 100
    build_strategy = fluid.BuildStrategy()
    if cfg.TRAIN.SYNC_BATCH_NORM and args.use_gpu:
        if dev_count > 1:
            # Apply sync batch norm strategy
            print("Sync BatchNorm strategy is effective.")
            build_strategy.sync_batch_norm = True
        else:
            print("Sync BatchNorm strategy will not be effective if GPU device"
                  " count <= 1")
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_loss.name,
        exec_strategy=exec_strategy,
        build_strategy=build_strategy)

    # Resume training
    begin_epoch = cfg.SOLVER.BEGIN_EPOCH
    if cfg.TRAIN.RESUME:
        begin_epoch = load_checkpoint(exe, train_prog)
    # Load pretrained model
    elif os.path.exists(cfg.TRAIN.PRETRAINED_MODEL):
        print('Pretrained model dir:', cfg.TRAIN.PRETRAINED_MODEL)
        load_vars = []

        def var_shape_matched(var, shape):
            """
            Check whehter persitable variable shape is match with current network
            """
            var_exist = os.path.exists(
                os.path.join(cfg.TRAIN.PRETRAINED_MODEL, var.name))
            if var_exist:
                var_shape = parse_shape_from_file(
                    os.path.join(cfg.TRAIN.PRETRAINED_MODEL, var.name))
                if var_shape == shape:
                    return True
                else:
                    print(
                        "Variable[{}] shape does not match current network, skip"
                        " to load it.".format(var.name))
                    return False

        for x in train_prog.list_vars():
            if isinstance(x, fluid.framework.Parameter):
                shape = tuple(fluid.global_scope().find_var(
                    x.name).get_tensor().shape())
                if var_shape_matched(x, shape):
                    load_vars.append(x)
        if cfg.MODEL.FP16:
            # If open FP16 training mode, load FP16 var separate
            load_fp16_vars(exe, cfg.TRAIN.PRETRAINED_MODEL, train_prog)
        else:
            fluid.io.load_vars(
                exe, dirname=cfg.TRAIN.PRETRAINED_MODEL, vars=load_vars)
        print("Pretrained model loaded successfully!")
    else:
        print('Pretrained model dir {} not exists, training from scratch...'.
              format(cfg.TRAIN.PRETRAINED_MODEL))

    fetch_list = [avg_loss.name, lr.name]
    if args.debug:
        # Fetch more variable info and use streaming confusion matrix to
        # calculate IoU results if in debug mode
        np.set_printoptions(
            precision=4, suppress=True, linewidth=160, floatmode="fixed")
        fetch_list.extend([pred.name, grts.name, masks.name])
        cm = ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=True)

    if args.use_tb:
        if not args.tb_log_dir:
            print("Please specify the log directory by --tb_log_dir.")
            exit(1)

        from tb_paddle import SummaryWriter

        if os.path.exists(args.tb_log_dir):
            shutil.rmtree(args.tb_log_dir)
        log_writer = SummaryWriter(args.tb_log_dir)

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
        print("Use multiprocess reader")
    else:
        print("Use multi-thread reader")

    for epoch in range(begin_epoch, cfg.SOLVER.NUM_EPOCHS + 1):
        py_reader.start()
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

                        print((
                            "epoch={} step={} lr={:.5f} loss={:.4f} acc={:.5f} mIoU={:.5f} step/sec={:.3f} | ETA {}"
                        ).format(epoch, global_step, lr[0], avg_loss, mean_acc,
                                 mean_iou, speed,
                                 calculate_eta(all_step - global_step, speed)))
                        print("Category IoU:", category_iou)
                        print("Category Acc:", category_acc)
                        if args.use_tb:
                            log_writer.add_scalar('Train/mean_iou', mean_iou,
                                                  global_step)
                            log_writer.add_scalar('Train/mean_acc', mean_acc,
                                                  global_step)
                            log_writer.add_scalar('Train/loss', avg_loss,
                                                  global_step)
                            log_writer.add_scalar('Train/lr', lr[0],
                                                  global_step)
                            log_writer.add_scalar('Train/step/sec', speed,
                                                  global_step)
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

                    if global_step % args.log_steps == 0:
                        avg_loss /= args.log_steps
                        speed = args.log_steps / timer.elapsed_time()
                        print((
                            "epoch={} step={} lr={:.5f} loss={:.4f} step/sec={:.3f} | ETA {}"
                        ).format(epoch, global_step, lr[0], avg_loss, speed,
                                 calculate_eta(all_step - global_step, speed)))
                        if args.use_tb:
                            log_writer.add_scalar('Train/loss', avg_loss,
                                                  global_step)
                            log_writer.add_scalar('Train/lr', lr[0],
                                                  global_step)
                            log_writer.add_scalar('Train/speed', speed,
                                                  global_step)
                        sys.stdout.flush()
                        avg_loss = 0.0
                        timer.restart()

            except fluid.core.EOFException:
                py_reader.reset()
                break
            except Exception as e:
                print(e)

        if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0:
            ckpt_dir = save_checkpoint(exe, train_prog, epoch)

            if args.do_eval:
                print("Evaluation start")
                _, mean_iou, _, mean_acc = evaluate(
                    cfg=cfg,
                    ckpt_dir=ckpt_dir,
                    use_gpu=args.use_gpu,
                    use_mpio=args.use_mpio)
                if args.use_tb:
                    log_writer.add_scalar('Evaluate/mean_iou', mean_iou,
                                          global_step)
                    log_writer.add_scalar('Evaluate/mean_acc', mean_acc,
                                          global_step)

            # Use Tensorboard to visualize results
            if args.use_tb and cfg.DATASET.VIS_FILE_LIST is not None:
                visualize(
                    cfg=cfg,
                    use_gpu=args.use_gpu,
                    vis_file_list=cfg.DATASET.VIS_FILE_LIST,
                    vis_dir="visual",
                    ckpt_dir=ckpt_dir,
                    log_writer=log_writer)

    # save final model
    save_checkpoint(exe, train_prog, 'final')


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer(reset_dataset=True)
    print(pprint.pformat(cfg))
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
