# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import functools
from collections import deque

import paddle
import paddle.nn.functional as F

import paddleseg.transforms.functional as Func
from paddleseg.utils import TimeAverager, calculate_eta, resume, logger, worker_init_fn
from paddleseg.models import losses
from models import EMA
from script import val
from utils import augmentation, load_ema_model, save_edge

paddle.set_printoptions(precision=15)


class Trainer():
    def __init__(self, model, cfg):
        '''model（nn.Layer): A sementic segmentation model.'''
        self.cfg = cfg
        self.ema_decay = cfg['ema_decay']
        self.edgeconstrain = cfg['edgeconstrain']
        self.edgepullin = cfg['edgepullin']
        self.src_only = cfg['src_only']
        self.featurepullin = cfg['featurepullin']

        self.model = model
        self.ema = EMA(self.model, self.ema_decay)
        self.celoss = losses.CrossEntropyLoss()
        self.klloss = losses.KLLoss()
        self.mseloss = losses.MSELoss()
        self.bceloss_src = losses.BCELoss(weight='dynamic')
        self.bceloss_tgt = losses.BCELoss(weight='dynamic')
        self.src_centers = [paddle.zeros((1, 19)) for _ in range(19)]
        self.tgt_centers = [paddle.zeros((1, 19)) for _ in range(19)]

        if 'None' in cfg['resume_ema']:
            self.resume_ema = None
        else:
            self.resume_ema = cfg['resume_ema']

    def train(self,
              train_dataset_src,
              train_dataset_tgt,
              val_dataset_tgt=None,
              val_dataset_src=None,
              optimizer=None,
              save_dir='output',
              iters=10000,
              batch_size=2,
              resume_model=None,
              save_interval=1000,
              log_iters=10,
              num_workers=0,
              use_vdl=False,
              keep_checkpoint_max=5,
              test_config=None):
        """
        Launch training.

        Args:
            train_dataset (paddle.io.Dataset): Used to read and process training datasets.
            val_dataset_tgt (paddle.io.Dataset, optional): Used to read and process validation datasets.
            optimizer (paddle.optimizer.Optimizer): The optimizer.
            save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
            iters (int, optional): How may iters to train the model. Defualt: 10000.
            batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
            resume_model (str, optional): The path of resume model.
            save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
            log_iters (int, optional): Display logging information at every log_iters. Default: 10.
            num_workers (int, optional): Num workers for data loader. Default: 0.
            use_vdl (bool, optional): Whether to record the data to VisualDL during training. Default: False.
            keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
            test_config(dict, optional): Evaluation config.
        """
        start_iter = 0
        self.model.train()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank

        if resume_model is not None:
            logger.info(resume_model)
            start_iter = resume(self.model, optimizer, resume_model)
        load_ema_model(self.model, self.resume_ema)

        if not os.path.isdir(save_dir):
            if os.path.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)

        if nranks > 1:
            paddle.distributed.fleet.init(is_collective=True)
            optimizer = paddle.distributed.fleet.distributed_optimizer(
                optimizer)  # The return is Fleet object
            ddp_model = paddle.distributed.fleet.distributed_model(self.model)

        batch_sampler_src = paddle.io.DistributedBatchSampler(
            train_dataset_src,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        loader_src = paddle.io.DataLoader(
            train_dataset_src,
            batch_sampler=batch_sampler_src,
            num_workers=num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn,
        )
        batch_sampler_tgt = paddle.io.DistributedBatchSampler(
            train_dataset_tgt,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        loader_tgt = paddle.io.DataLoader(
            train_dataset_tgt,
            batch_sampler=batch_sampler_tgt,
            num_workers=num_workers,
            return_list=True,
            worker_init_fn=worker_init_fn,
        )

        if use_vdl:
            from visualdl import LogWriter
            log_writer = LogWriter(save_dir)

        iters_per_epoch = len(batch_sampler_tgt)
        best_mean_iou = -1.0
        best_model_iter = -1
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        save_models = deque()
        batch_start = time.time()

        iter = start_iter
        while iter < iters:
            for _, (data_src, data_tgt) in enumerate(
                    zip(loader_src, loader_tgt)):

                reader_cost_averager.record(time.time() - batch_start)
                loss_dict = {}

                #### training #####
                images_tgt = data_tgt[0]
                labels_tgt = data_tgt[1].astype('int64')
                images_src = data_src[0]
                labels_src = data_src[1].astype('int64')

                edges_src = data_src[2].astype('int64')
                edges_tgt = data_tgt[2].astype('int64')

                if nranks > 1:
                    logits_list_src = ddp_model(images_src)
                else:
                    logits_list_src = self.model(images_src)

                ##### source seg & edge loss ####
                loss_src_seg_main = self.celoss(logits_list_src[0], labels_src)
                loss_src_seg_aux = 0.1 * self.celoss(logits_list_src[1],
                                                     labels_src)

                loss_src_seg = loss_src_seg_main + loss_src_seg_aux
                loss_dict["source_main"] = loss_src_seg_main.numpy()[0]
                loss_dict["source_aux"] = loss_src_seg_aux.numpy()[0]
                loss = loss_src_seg
                del loss_src_seg, loss_src_seg_aux, loss_src_seg_main

                #### generate target pseudo label  ####
                with paddle.no_grad():
                    if nranks > 1:
                        logits_list_tgt = ddp_model(images_tgt)
                    else:
                        logits_list_tgt = self.model(images_tgt)

                    pred_P_1 = F.softmax(logits_list_tgt[0], axis=1)
                    labels_tgt_psu = paddle.argmax(pred_P_1.detach(), axis=1)

                    # aux label
                    pred_P_2 = F.softmax(logits_list_tgt[1], axis=1)
                    pred_c = (pred_P_1 + pred_P_2) / 2
                    labels_tgt_psu_aux = paddle.argmax(pred_c.detach(), axis=1)

                if self.edgeconstrain:
                    loss_src_edge = self.bceloss_src(
                        logits_list_src[2], edges_src)  # 1, 2 640, 1280
                    src_edge = paddle.argmax(
                        logits_list_src[2].detach().clone(),
                        axis=1)  # 1, 1, 640,1280
                    src_edge_acc = ((src_edge == edges_src).numpy().sum().astype('float32')\
                                                /functools.reduce(lambda a, b: a * b, src_edge.shape))*100

                    if (not self.src_only) and (iter > 200000):
                        ####  target seg & edge loss ####
                        logger.info("Add target edege loss")
                        edges_tgt = Func.mask_to_binary_edge(
                            labels_tgt_psu.detach().clone().numpy(),
                            radius=2,
                            num_classes=train_dataset_tgt.NUM_CLASSES)
                        edges_tgt = paddle.to_tensor(edges_tgt, dtype='int64')

                        loss_tgt_edge = self.bceloss_tgt(
                            logits_list_tgt[2], edges_tgt)
                        loss_edge = loss_tgt_edge + loss_src_edge
                    else:
                        loss_tgt_edge = paddle.zeros([1])
                        loss_edge = loss_src_edge

                    loss += loss_edge

                    loss_dict['target_edge'] = loss_tgt_edge.numpy()[0]
                    loss_dict['source_edge'] = loss_src_edge.numpy()[0]

                    del loss_edge, loss_tgt_edge, loss_src_edge

                #### target aug loss #######
                augs = augmentation.get_augmentation()
                images_tgt_aug, labels_tgt_aug = augmentation.augment(
                    images=images_tgt.cpu(),
                    labels=labels_tgt_psu.detach().cpu(),
                    aug=augs,
                    iters="{}_1".format(iter))
                images_tgt_aug = images_tgt_aug.cuda()
                labels_tgt_aug = labels_tgt_aug.cuda()

                _, labels_tgt_aug_aux = augmentation.augment(
                    images=images_tgt.cpu(),
                    labels=labels_tgt_psu_aux.detach().cpu(),
                    aug=augs,
                    iters="{}_2".format(iter))
                labels_tgt_aug_aux = labels_tgt_aug_aux.cuda()

                if nranks > 1:
                    logits_list_tgt_aug = ddp_model(images_tgt_aug)
                else:
                    logits_list_tgt_aug = self.model(images_tgt_aug)

                loss_tgt_aug_main = 0.1 * (self.celoss(logits_list_tgt_aug[0],
                                                       labels_tgt_aug))
                loss_tgt_aug_aux = 0.1 * (0.1 * self.celoss(
                    logits_list_tgt_aug[1], labels_tgt_aug_aux))

                loss_tgt_aug = loss_tgt_aug_aux + loss_tgt_aug_main

                loss += loss_tgt_aug

                loss_dict['target_aug_main'] = loss_tgt_aug_main.numpy()[0]
                loss_dict['target_aug_aux'] = loss_tgt_aug_aux.numpy()[0]
                del images_tgt_aug, labels_tgt_aug_aux, images_tgt, \
                    loss_tgt_aug, loss_tgt_aug_aux, loss_tgt_aug_main

                #### edge input seg; src & tgt edge pull in ######
                if self.edgepullin:
                    src_edge_logit = logits_list_src[2]
                    feat_src = paddle.concat(
                        [logits_list_src[0], src_edge_logit], axis=1).detach()

                    out_src = self.model.fusion(feat_src)
                    loss_src_edge_rec = self.celoss(out_src, labels_src)

                    tgt_edge_logit = logits_list_tgt_aug[2]
                    # tgt_edge_logit = paddle.to_tensor(
                    #     Func.mask_to_onehot(edges_tgt.squeeze().numpy(), 2)
                    #     ).unsqueeze(0).astype('float32')
                    feat_tgt = paddle.concat(
                        [logits_list_tgt[0], tgt_edge_logit], axis=1).detach()

                    out_tgt = self.model.fusion(feat_tgt)
                    loss_tgt_edge_rec = self.celoss(out_tgt, labels_tgt)

                    loss_edge_rec = loss_tgt_edge_rec + loss_src_edge_rec
                    loss += loss_edge_rec

                    loss_dict['src_edge_rec'] = loss_src_edge_rec.numpy()[0]
                    loss_dict['tgt_edge_rec'] = loss_tgt_edge_rec.numpy()[0]

                    del loss_tgt_edge_rec, loss_src_edge_rec

                #### mask input feature & pullin  ######
                if self.featurepullin:
                    # inner-class loss
                    feat_src = logits_list_src[0]
                    feat_tgt = logits_list_tgt_aug[0]
                    center_src_s, center_tgt_s = [], []

                    total_pixs = logits_list_src[0].shape[2] * \
                                    logits_list_src[0].shape[3]

                    for i in range(train_dataset_tgt.NUM_CLASSES):
                        pred = paddle.argmax(
                            logits_list_src[0].detach().clone(),
                            axis=1).unsqueeze(0)  # 1, 1, 640, 1280
                        sel_num = paddle.sum((pred == i).astype('float32'))
                        # ignore tensor that do not have features in this img
                        if sel_num > 0:
                            feat_sel_src = paddle.where(
                                (pred == i).expand_as(feat_src), feat_src,
                                paddle.zeros(feat_src.shape))
                            center_src = paddle.mean(
                                feat_sel_src, axis=[2, 3]) / (
                                    sel_num / total_pixs)  # 1, C

                            self.src_centers[i] = 0.99 * self.src_centers[i] + (
                                1 - 0.99) * center_src

                        pred = labels_tgt_aug.unsqueeze(0)  # 1, 1,  512, 512
                        sel_num = paddle.sum((pred == i).astype('float32'))
                        if sel_num > 0:
                            feat_sel_tgt = paddle.where(
                                (pred == i).expand_as(feat_tgt), feat_tgt,
                                paddle.zeros(feat_tgt.shape))
                            center_tgt = paddle.mean(
                                feat_sel_tgt, axis=[2, 3
                                                    ]) / (sel_num / total_pixs)

                            self.tgt_centers[i] = 0.99 * self.tgt_centers[i] + (
                                1 - 0.99) * center_tgt
                        center_src_s.append(center_src)
                        center_tgt_s.append(center_tgt)

                    if iter >= 3000:  # average center structure alignment
                        src_centers = paddle.concat(self.src_centers, axis=0)
                        tgt_centers = paddle.concat(
                            self.tgt_centers, axis=0)  # 19， 2048

                        relatmat_src = paddle.matmul(
                            src_centers, src_centers, transpose_y=True)  # 19，19
                        relatmat_tgt = paddle.matmul(
                            tgt_centers, tgt_centers, transpose_y=True)

                        loss_intra_relate = self.klloss(relatmat_src, (relatmat_tgt+relatmat_src)/2) \
                                            + self.klloss(relatmat_tgt, (relatmat_tgt+relatmat_src)/2)

                        loss_pix_align_src = self.mseloss(
                            paddle.to_tensor(center_src_s),
                            paddle.to_tensor(self.src_centers).detach().clone())
                        loss_pix_align_tgt = self.mseloss(
                            paddle.to_tensor(center_tgt_s),
                            paddle.to_tensor(self.tgt_centers).detach().clone())

                        loss_feat_align = loss_pix_align_src + loss_pix_align_tgt + loss_intra_relate

                        loss += loss_feat_align

                        loss_dict['loss_pix_align_src'] = \
                                loss_pix_align_src.numpy()[0]
                        loss_dict['loss_pix_align_tgt'] = \
                                loss_pix_align_tgt.numpy()[0]
                        loss_dict['loss_intra_relate'] = \
                                loss_intra_relate.numpy()[0]

                        del loss_pix_align_tgt, loss_pix_align_src, loss_intra_relate,

                    self.tgt_centers = [
                        item.detach().clone() for item in self.tgt_centers
                    ]
                    self.src_centers = [
                        item.detach().clone() for item in self.src_centers
                    ]

                loss.backward()
                del loss
                loss = sum(loss_dict.values())

                optimizer.step()
                self.ema.update_params()

                with paddle.no_grad():
                    ##### log & save #####
                    lr = optimizer.get_lr()
                    # update lr
                    if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                        lr_sche = optimizer.user_defined_optimizer._learning_rate
                    else:
                        lr_sche = optimizer._learning_rate
                    if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                        lr_sche.step()

                    if self.cfg['save_edge']:
                        tgt_edge = paddle.argmax(
                            logits_list_tgt_aug[2].detach().clone(),
                            axis=1)  # 1, 1, 640,1280
                        src_feed_gt = paddle.argmax(
                            src_edge_logit.astype('float32'), axis=1)
                        tgt_feed_gt = paddle.argmax(
                            tgt_edge_logit.astype('float32'), axis=1)
                        logger.info('src_feed_gt_{}_{}_{}'.format(
                            src_feed_gt.shape, src_feed_gt.max(),
                            src_feed_gt.min()))
                        logger.info('tgt_feed_gt_{}_{}_{}'.format(
                            tgt_feed_gt.shape, max(tgt_feed_gt),
                            min(tgt_feed_gt)))
                        save_edge(src_feed_gt, 'src_feed_gt_{}'.format(iter))
                        save_edge(tgt_feed_gt, 'tgt_feed_gt_{}'.format(iter))
                        save_edge(tgt_edge, 'tgt_pred_{}'.format(iter))
                        save_edge(src_edge, 'src_pred_{}_{}'.format(
                            iter, src_edge_acc))
                        save_edge(edges_src, 'src_gt_{}'.format(iter))
                        save_edge(edges_tgt, 'tgt_gt_{}'.format(iter))

                    self.model.clear_gradients()

                    batch_cost_averager.record(
                        time.time() - batch_start, num_samples=batch_size)

                    iter += 1
                    if (iter) % log_iters == 0 and local_rank == 0:
                        label_tgt_acc = ((labels_tgt == labels_tgt_psu).numpy().sum().astype('float32')\
                                            /functools.reduce(lambda a, b: a * b, labels_tgt_psu.shape))*100

                        remain_iters = iters - iter
                        avg_train_batch_cost = batch_cost_averager.get_average()
                        avg_train_reader_cost = reader_cost_averager.get_average(
                        )
                        eta = calculate_eta(remain_iters, avg_train_batch_cost)
                        logger.info(
                            "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, tgt_pix_acc: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                            .format((iter - 1) // iters_per_epoch + 1, iter,
                                    iters, loss, label_tgt_acc, lr,
                                    avg_train_batch_cost, avg_train_reader_cost,
                                    batch_cost_averager.get_ips_average(), eta))

                        if use_vdl:
                            log_writer.add_scalar('Train/loss', loss, iter)
                            # Record all losses if there are more than 2 losses.
                            if len(loss_dict) > 1:
                                for name, loss in loss_dict.items():
                                    log_writer.add_scalar(
                                        'Train/loss_' + name, loss, iter)

                            log_writer.add_scalar('Train/lr', lr, iter)
                            log_writer.add_scalar('Train/batch_cost',
                                                  avg_train_batch_cost, iter)
                            log_writer.add_scalar('Train/reader_cost',
                                                  avg_train_reader_cost, iter)
                            log_writer.add_scalar('Train/tgt_label_acc',
                                                  label_tgt_acc, iter)

                        reader_cost_averager.reset()
                        batch_cost_averager.reset()

                    if (iter % save_interval == 0
                            or iter == iters) and (val_dataset_tgt is not None):
                        num_workers = 4 if num_workers > 0 else 0  # adjust num_worker=4

                        if test_config is None:
                            test_config = {}
                        self.ema.apply_shadow()
                        self.ema.model.eval()

                        PA_tgt, _, MIoU_tgt, _ = val.evaluate(
                            self.model,
                            val_dataset_tgt,
                            num_workers=num_workers,
                            **test_config)

                        if (iter % (save_interval * 30)) == 0 \
                            and self.cfg['eval_src']:  # add evaluate on src
                            PA_src, _, MIoU_src, _ = val.evaluate(
                                self.model,
                                val_dataset_src,
                                num_workers=num_workers,
                                **test_config)
                            logger.info(
                                '[EVAL] The source mIoU is ({:.4f}) at iter {}.'
                                .format(MIoU_src, iter))

                        self.ema.restore()
                        self.model.train()

                    if (iter % save_interval == 0
                            or iter == iters) and local_rank == 0:
                        current_save_dir = os.path.join(save_dir,
                                                        "iter_{}".format(iter))
                        if not os.path.isdir(current_save_dir):
                            os.makedirs(current_save_dir)
                        paddle.save(
                            self.model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                        paddle.save(
                            self.ema.shadow,
                            os.path.join(current_save_dir,
                                         'model_ema.pdparams'))
                        paddle.save(
                            optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                        save_models.append(current_save_dir)
                        if len(save_models) > keep_checkpoint_max > 0:
                            model_to_remove = save_models.popleft()
                            shutil.rmtree(model_to_remove)

                        if val_dataset_tgt is not None:
                            if MIoU_tgt > best_mean_iou:
                                best_mean_iou = MIoU_tgt
                                best_model_iter = iter
                                best_model_dir = os.path.join(
                                    save_dir, "best_model")
                                paddle.save(
                                    self.model.state_dict(),
                                    os.path.join(best_model_dir,
                                                 'model.pdparams'))

                            logger.info(
                                '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                                .format(best_mean_iou, best_model_iter))

                        if use_vdl:
                            log_writer.add_scalar('Evaluate/mIoU', MIoU_tgt,
                                                  iter)
                            log_writer.add_scalar('Evaluate/PA', PA_tgt, iter)

                            if self.cfg['eval_src']:
                                log_writer.add_scalar('Evaluate/mIoU_src',
                                                      MIoU_src, iter)
                                log_writer.add_scalar('Evaluate/PA_src', PA_src,
                                                      iter)

                    batch_start = time.time()

            self.ema.update_buffer()

        # # Calculate flops.
        if local_rank == 0:

            def count_syncbn(m, x, y):
                x = x[0]
                nelements = x.numel()
                m.total_ops += int(2 * nelements)

            _, c, h, w = images_src.shape
            flops = paddle.flops(
                self.model, [1, c, h, w],
                custom_ops={paddle.nn.SyncBatchNorm: count_syncbn})

        # Sleep for half a second to let dataloader release resources.
        time.sleep(0.5)
        if use_vdl:
            log_writer.close()
