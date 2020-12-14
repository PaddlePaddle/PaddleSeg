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

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(cur_path)[0])[0]
SEG_PATH = os.path.join(cur_path, "../../../")
sys.path.append(SEG_PATH)
sys.path.append(root_path)

import argparse
import pprint

import numpy as np
import paddle.fluid as fluid

from utils.config import cfg
from pdseg.utils.timer import Timer, calculate_eta
from reader import LaneNetDataset
from models.model_builder import build_model
from models.model_builder import ModelPhase
from eval import evaluate
from vis import visualize
from utils import dist_utils
from utils.load_model_utils import load_pretrained_weights


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
    return parser.parse_args()


def save_checkpoint(exe, program, ckpt_name):
    """
    Save checkpoint for evaluation or resume training
    """
    ckpt_dir = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, str(ckpt_name))
    print("Save model checkpoint to {}".format(ckpt_dir))
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    fluid.save(program, os.path.join(ckpt_dir, 'model'))

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
    fluid.load(program, os.path.join(model_path, 'model'), exe)

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


def print_info(*msg):
    if cfg.TRAINER_ID == 0:
        print(*msg)


def train(cfg):
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    drop_last = True

    dataset = LaneNetDataset(
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
                    yield item
                batch_data = []

    # Get device environment
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
    cfg.BATCH_SIZE_PER_DEV = batch_size_per_dev
    print_info("batch_size_per_dev: {}".format(batch_size_per_dev))

    data_loader, avg_loss, lr, pred, grts, masks, emb_loss, seg_loss, accuracy, fp, fn = build_model(
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
        load_pretrained_weights(exe, train_prog, cfg.TRAIN.PRETRAINED_MODEL_DIR)
    else:
        print_info(
            'Pretrained model dir {} not exists, training from scratch...'.
            format(cfg.TRAIN.PRETRAINED_MODEL_DIR))

    # fetch_list = [avg_loss.name, lr.name, accuracy.name, precision.name, recall.name]
    fetch_list = [
        avg_loss.name, lr.name, seg_loss.name, emb_loss.name, accuracy.name,
        fp.name, fn.name
    ]
    if args.debug:
        # Fetch more variable info and use streaming confusion matrix to
        # calculate IoU results if in debug mode
        np.set_printoptions(
            precision=4, suppress=True, linewidth=160, floatmode="fixed")
        fetch_list.extend([pred.name, grts.name, masks.name])
        # cm = ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=True)

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
    avg_seg_loss = 0.0
    avg_emb_loss = 0.0
    avg_acc = 0.0
    avg_fp = 0.0
    avg_fn = 0.0
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
                # If not in debug mode, avoid unnessary log and calculate
                loss, lr, out_seg_loss, out_emb_loss, out_acc, out_fp, out_fn = exe.run(
                    program=compiled_train_prog,
                    fetch_list=fetch_list,
                    return_numpy=True)

                avg_loss += np.mean(np.array(loss))
                avg_seg_loss += np.mean(np.array(out_seg_loss))
                avg_emb_loss += np.mean(np.array(out_emb_loss))
                avg_acc += np.mean(out_acc)
                avg_fp += np.mean(out_fp)
                avg_fn += np.mean(out_fn)
                step += 1

                if step % args.log_steps == 0 and cfg.TRAINER_ID == 0:
                    avg_loss /= args.log_steps
                    avg_seg_loss /= args.log_steps
                    avg_emb_loss /= args.log_steps
                    avg_acc /= args.log_steps
                    avg_fp /= args.log_steps
                    avg_fn /= args.log_steps
                    speed = args.log_steps / timer.elapsed_time()
                    print((
                        "epoch={} step={} lr={:.5f} loss={:.4f} seg_loss={:.4f} emb_loss={:.4f} accuracy={:.4} fp={:.4} fn={:.4} step/sec={:.3f} | ETA {}"
                    ).format(epoch, step, lr[0], avg_loss, avg_seg_loss,
                             avg_emb_loss, avg_acc, avg_fp, avg_fn, speed,
                             calculate_eta(all_step - step, speed)))
                    if args.use_vdl:
                        log_writer.add_scalar('Train/loss', avg_loss, step)
                        log_writer.add_scalar('Train/lr', lr[0], step)
                        log_writer.add_scalar('Train/speed', speed, step)
                    sys.stdout.flush()
                    avg_loss = 0.0
                    avg_seg_loss = 0.0
                    avg_emb_loss = 0.0
                    avg_acc = 0.0
                    avg_fp = 0.0
                    avg_fn = 0.0
                    timer.restart()

            except fluid.core.EOFException:
                data_loader.reset()
                break
            except Exception as e:
                print(e)

        if epoch % cfg.TRAIN.SNAPSHOT_EPOCH == 0 and cfg.TRAINER_ID == 0:
            ckpt_dir = save_checkpoint(exe, train_prog, epoch)

            if args.do_eval:
                print("Evaluation start")
                accuracy, fp, fn = evaluate(
                    cfg=cfg,
                    ckpt_dir=ckpt_dir,
                    use_gpu=args.use_gpu,
                    use_mpio=args.use_mpio)
                if args.use_vdl:
                    log_writer.add_scalar('Evaluate/accuracy', accuracy, step)
                    log_writer.add_scalar('Evaluate/fp', fp, step)
                    log_writer.add_scalar('Evaluate/fn', fn, step)

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
        save_checkpoint(exe, train_prog, 'final')


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)

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
