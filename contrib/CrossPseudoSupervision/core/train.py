# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from collections import deque
import shutil

import paddle
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler, DataLoader
from paddleseg.utils import (TimeAverager, calculate_eta, logger,
                             train_profiler, op_flops_funs)
from paddleseg.core.val import evaluate

from utils import cps_resume
from batch_transforms import SegCollate, AddMaskParamsToBatch


def get_train_loader(nranks,
                     train_dataset,
                     batch_size,
                     num_workers,
                     collate_fn=None):
    train_sampler = None
    is_shuffle = True
    batch_size = batch_size

    if nranks > 1:
        is_shuffle = False
        batch_size_per_gpu = int(batch_size / nranks)
        train_sampler = DistributedBatchSampler(
            train_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=is_shuffle,
            drop_last=True)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            use_shared_memory=True,
            collate_fn=collate_fn)
    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            shuffle=is_shuffle,
            use_shared_memory=True,
            collate_fn=collate_fn)

    return train_loader


def cps_loss_computation(nranks, logits_cons_stu_1, logits_cons_stu_2,
                         sup_pred_l, sup_pred_r, ps_label_1, ps_label_2, gts,
                         losses):
    # cps loss
    cps_loss = losses['types'][0](
        logits_cons_stu_1, ps_label_2) + losses['types'][0](logits_cons_stu_2,
                                                            ps_label_1)
    if nranks > 1:
        dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
        cps_loss = cps_loss / nranks
    cps_loss = cps_loss * losses['coef'][0]

    # sup loss for sub network 1
    loss_sup = losses['types'][1](sup_pred_l, gts)
    if nranks > 1:
        dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
        loss_sup = loss_sup / nranks
    loss_sup = loss_sup * losses['coef'][1]

    # sup loss for sub network 2
    loss_sup_r = losses['types'][2](sup_pred_r, gts)
    if nranks > 1:
        dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
        loss_sup_r = loss_sup_r / nranks
    loss_sup_r = loss_sup_r * losses['coef'][2]

    loss_list = [cps_loss, loss_sup, loss_sup_r]

    return loss_list


def train(model,
          train_dataset,
          unsupervised_train_dataset,
          mask_genarator,
          val_dataset=None,
          optimizer_l=None,
          optimizer_r=None,
          save_dir='output',
          nepochs=240,
          labeled_ratio=2,
          batch_size=8,
          resume_model=None,
          save_epoch=5,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None,
          keep_checkpoint_max=5,
          test_config=None,
          profiler_options=None):
    """
    Launch training.

    Args:
        model (nn.Layer): A semantic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        mask_generator (batch_transforms.mask_gen.BoxMaskGenerator): Cutmix used for training.
        unsupervised_train_dataset (paddle.io.Dataset, optional): Used to read and process training datasets do not have labels.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer_l (paddle.optimizer.Optimizer, optional): The optimizer for the first sub-model.
        optimizer_r (paddle.optimizer.Optimizer, optional): The optimizer for the second sub-model.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        nepochs (int, optional): How may epochs to train the model. Defualt: 240.
        labeled_ratio (int, optional): The ratio of total data to marked data. If 2, we use the ratio of 1/2, i.e. 0.5. Default: 2. 
        batch_size (int, optional): Mini batch size of total gpus or cpu. Default: 8.
        resume_model (str, optional): The path of resume model.
        save_epoch (int, optional): How many epochs to save a model snapshot once during training. Default: 5.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
        losses (dict, optional): A dict including 'types' and 'coef'. The length of coef should equal to 1 or len(losses['types']).
            The 'types' item is a list of object of paddleseg.models.losses while the 'coef' item is a list of the relevant coefficient.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
        test_config(dict, optional): Evaluation config.
        profiler_options (str, optional): The option of train profiler.
    """
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    total_imgs = len(train_dataset) + len(unsupervised_train_dataset)
    num_train_imgs = total_imgs // labeled_ratio
    num_unsup_imgs = total_imgs - num_train_imgs
    max_samples = max(num_train_imgs, num_unsup_imgs)
    niters_per_epoch = max_samples // batch_size

    start_epoch = 0
    if resume_model is not None:
        start_epoch = cps_resume(model, optimizer_l, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    raw_model = model  # use for eval during training

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer_l = paddle.distributed.fleet.distributed_optimizer(
            optimizer_l)  # The return is Fleet object
        optimizer_r = paddle.distributed.fleet.distributed_optimizer(
            optimizer_r)  # The return is Fleet object
        model = paddle.distributed.fleet.distributed_model(raw_model)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_genarator)
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    train_loader = get_train_loader(
        nranks, train_dataset, batch_size, num_workers, collate_fn=collate_fn)
    unsupervised_train_loader_0 = get_train_loader(nranks, unsupervised_train_dataset, batch_size, \
                                                   num_workers, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1 = get_train_loader(nranks, unsupervised_train_dataset, batch_size, \
                                                   num_workers, collate_fn=collate_fn)

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    avg_loss = 0.0
    avg_loss_list = []
    best_mean_iou = -1.0
    best_model_epoch = -1
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    total_iters = niters_per_epoch * nepochs

    for epoch in range(start_epoch, nepochs):
        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        for idx in range(niters_per_epoch):
            optimizer_l.clear_grad()
            optimizer_r.clear_grad()

            data = dataloader.__next__()
            unsup_data_0 = unsupervised_dataloader_0.__next__()
            unsup_data_1 = unsupervised_dataloader_1.__next__()

            imgs = data['img']
            gts = data['label'].astype('int64')
            unsup_imgs_0 = unsup_data_0['img']
            unsup_imgs_1 = unsup_data_1['img']
            mask_params = unsup_data_0['mask_params']

            if hasattr(model, 'data_format') and model.data_format == 'NHWC':
                imgs = imgs.transpose((0, 2, 3, 1))
                unsup_imgs_0 = unsup_imgs_0.transpose((0, 2, 3, 1))
                unsup_imgs_1 = unsup_imgs_1.transpose((0, 2, 3, 1))
                mask_params = mask_params.transpose((0, 2, 3, 1))

            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (
                1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

            with paddle.no_grad():
                # Estimate the pseudo-label with one branch & supervise another branch
                logits_u0_tea_1, logits_u0_tea_2 = model(unsup_imgs_0)
                logits_u1_tea_1, logits_u1_tea_2 = model(unsup_imgs_1)
                logits_u0_tea_1 = logits_u0_tea_1[0].detach()
                logits_u1_tea_1 = logits_u1_tea_1[0].detach()
                logits_u0_tea_2 = logits_u0_tea_2[0].detach()
                logits_u1_tea_2 = logits_u1_tea_2[0].detach()

            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * (
                1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            ps_label_1 = paddle.argmax(logits_cons_tea_1, axis=1)
            logits_cons_tea_2 = logits_u0_tea_2 * (
                1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            ps_label_2 = paddle.argmax(logits_cons_tea_2, axis=1)

            logits_cons_stu_1, logits_cons_stu_2 = model(unsup_imgs_mixed)
            sup_pred_l, sup_pred_r = model(imgs)

            logits_cons_stu_1, logits_cons_stu_2 = logits_cons_stu_1[
                0], logits_cons_stu_2[0]
            sup_pred_l, sup_pred_r = sup_pred_l[0], sup_pred_r[0]

            loss_list = cps_loss_computation(nranks, logits_cons_stu_1, logits_cons_stu_2, sup_pred_l, sup_pred_r, \
                                 ps_label_1, ps_label_2, gts, losses)
            loss = sum(loss_list)
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()
            lr = optimizer_l.get_lr()

            # update lr
            if isinstance(optimizer_l, paddle.distributed.fleet.Fleet):
                lr_sche_l = optimizer_l.user_defined_optimizer._learning_rate
            else:
                lr_sche_l = optimizer_l._learning_rate
            if isinstance(optimizer_r, paddle.distributed.fleet.Fleet):
                lr_sche_r = optimizer_r.user_defined_optimizer._learning_rate
            else:
                lr_sche_r = optimizer_r._learning_rate

            if isinstance(lr_sche_l, paddle.optimizer.lr.LRScheduler):
                if isinstance(lr_sche_l, paddle.optimizer.lr.ReduceOnPlateau):
                    lr_sche_l.step(loss)
                else:
                    lr_sche_l.step()
            if isinstance(lr_sche_r, paddle.optimizer.lr.LRScheduler):
                if isinstance(lr_sche_r, paddle.optimizer.lr.ReduceOnPlateau):
                    lr_sche_r.step(loss)
                else:
                    lr_sche_r.step()

            train_profiler.add_profiler_step(profiler_options)

            model.clear_gradients()
            avg_loss += float(loss)
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            current_iter = epoch * niters_per_epoch + idx + 1
            if ((current_iter % log_iters) == 0) and (local_rank == 0):
                avg_loss /= log_iters
                avg_loss_list = [l.item() / log_iters for l in avg_loss_list]
                remain_iters = total_iters - current_iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format(epoch + 1, current_iter, total_iters, avg_loss, lr,
                            avg_train_batch_cost, avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, current_iter)
                    # Record all losses if there are more than 1 loss.
                    if len(avg_loss_list) > 1:
                        avg_loss_dict = {}
                        for i, value in enumerate(avg_loss_list):
                            avg_loss_dict['loss_' + str(i)] = value
                        for key, value in avg_loss_dict.items():
                            log_tag = 'Train/' + key
                            log_writer.add_scalar(log_tag, value, current_iter)

                    log_writer.add_scalar('Train/lr', lr, current_iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, current_iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, current_iter)
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            batch_start = time.time()

        if ((epoch + 1) % save_epoch == 0 or
            (epoch + 1) == nepochs) and (val_dataset is not None) and (
                (epoch + 1) > (nepochs / 1.2)):

            num_workers = 1 if num_workers > 0 else 0

            if test_config is None:
                test_config = {}

            mean_iou, acc, _, _, _ = evaluate(
                raw_model, val_dataset, num_workers=num_workers, **test_config)

            raw_model.train()

        if ((epoch + 1) % save_epoch == 0 or
            (epoch + 1) == nepochs) and local_rank == 0 and (
                (epoch + 1) > (nepochs / 1.2)):
            current_save_dir = os.path.join(save_dir,
                                            "epoch_{}".format(epoch + 1))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)

            paddle.save(raw_model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer_l.state_dict(),
                        os.path.join(current_save_dir, 'model_l.pdopt'))
            paddle.save(optimizer_r.state_dict(),
                        os.path.join(current_save_dir, 'model_r.pdopt'))

            save_models.append(current_save_dir)
            if len(save_models) > keep_checkpoint_max > 0:
                model_to_remove = save_models.popleft()
                shutil.rmtree(model_to_remove)

            if val_dataset is not None:
                if mean_iou > best_mean_iou:
                    best_mean_iou = mean_iou
                    best_model_epoch = epoch + 1
                    best_model_dir = os.path.join(save_dir, "best_model")
                    paddle.save(raw_model.state_dict(),
                                os.path.join(best_model_dir, 'model.pdparams'))
                logger.info(
                    '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at epoch {}.'
                    .format(best_mean_iou, best_model_epoch))

                if use_vdl:
                    log_writer.add_scalar('Evaluate/mIoU', mean_iou, epoch + 1)
                    log_writer.add_scalar('Evaluate/Acc', acc, epoch + 1)

    # Sleep for a second to let dataloader release resources.
    time.sleep(1)
    if use_vdl:
        log_writer.close()
