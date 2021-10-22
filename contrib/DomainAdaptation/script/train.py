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
from collections import deque
import shutil

import paddle
import paddle.nn.functional as F

from paddleseg.utils import TimeAverager, calculate_eta, resume, logger, worker_init_fn
from paddleseg.models import losses
import transforms.functional as Func
from models import EMA
from script import val
from utils import augmentation
from models.losses import KLLoss

paddle.set_printoptions(precision=15)


class Trainer():
    def __init__(self, model, ema_decay):
        '''model（nn.Layer): A sementic segmentation model.'''
        self.model = model
        self.ema_decay = ema_decay
        # self.ema = EMA(self.model, ema_decay)
        self.celoss = losses.CrossEntropyLoss()
        self.klloss = KLLoss()
        self.bceloss = losses.BCELoss()

    def train(self,
              train_dataset_src,
              train_dataset_tgt,
              val_dataset_tgt=None,
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
              test_config=None,
              pseudolabel_threshold=0.0,
              edgeconstrain=False,
              edgepullin=False,
              featurepullin=False,
              reprod_logger=None):
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
            start_iter = resume(self.model, optimizer, resume_model)

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
        import numpy as np

        # reprod_logger.add("length_src_train", np.array(len(train_dataset_src)))
        # reprod_logger.add("length_tgt_train", np.array(len(train_dataset_tgt)))
        # reprod_logger.add("length_tgt_val", np.array(len(val_dataset_tgt)))

        # for idx in range(3):
        #     import random
        #     np.random.seed(0)
        #     random.seed(0)
        #     print('############dataeval')
        #     rnd_idx = [np.random.randint(0, len(train_dataset_tgt)) for i in range(5)]
        #     reprod_logger.add(f"dataset_src_{train_dataset_src[rnd_idx[idx]][2]}",
        #                         train_dataset_src[rnd_idx[idx]][0].numpy())
        #     reprod_logger.add(f"dataset_tgt_{train_dataset_tgt[rnd_idx[idx]][2]}",
        #                         train_dataset_tgt[rnd_idx[idx]][0].numpy())
        # print(rnd_idx)

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
                # print('iter', iter)
                # if iter == 0:
                #     data_src_0, data_tgt_0 = data_src, data_tgt

                loss_dict = {}
                reader_cost_averager.record(time.time() - batch_start)

                #### training #####
                images_tgt = data_tgt[0]
                labels_tgt = data_tgt[1].astype('int64')
                images_src = data_src[0]
                labels_src = data_src[1].astype('int64')

                edges_src = None
                if len(data_src) == 3:
                    edges_src = data_src[2].astype('int64')

                # images_src, labels_src = paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_data_src.npy')), \
                #             paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_label_src.npy')).astype('int64')
                # images_tgt, labels_tgt = paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_data_tgt.npy')), \
                #             paddle.to_tensor(np.load('/ssd2/tangshiyu/Code/fake_label_tgt.npy')).astype('int64')
                # reprod_logger.add("images_tgt_{}".format(data_tgt[2].numpy()[0]), images_tgt.cpu().detach().numpy())
                # reprod_logger.add("images_src_{}".format(data_src[2].numpy()[0]), images_src.cpu().detach().numpy())

                if nranks > 1:
                    # logits_list_tgt = ddp_model(images_tgt)
                    logits_list_src = ddp_model(images_src)
                else:
                    # logits_list_tgt = self.model(images_tgt)
                    logits_list_src = self.model(images_src)

                # res_src = logits_list_src[0] + logits_list_src[1]
                # reprod_logger.add("res_src_{}".format(iter), res_src.cpu().detach().numpy())

                #### target pseudo label  ####
                with paddle.no_grad():
                    if nranks > 1:
                        logits_list_tgt = ddp_model(images_tgt)
                    else:
                        logits_list_tgt = self.model(images_tgt)

                    pred_P_1 = F.softmax(logits_list_tgt[0], axis=1)
                    labels_tgt_psu = paddle.argmax(pred_P_1.detach(), axis=1)
                    # reprod_logger.add("labels_tgt_psu_{}".format(data_tgt[2].numpy()[0]), labels_tgt_psu.cpu().detach().numpy())

                    # aux label
                    pred_P_2 = F.softmax(logits_list_tgt[1], axis=1)
                    pred_c = (pred_P_1 + pred_P_2) / 2
                    labels_tgt_psu_aux = paddle.argmax(pred_c.detach(), axis=1)
                    # reprod_logger.add("labels_tgt_psu_aux_{}".format(data_tgt[2].numpy()[0]), labels_tgt_psu_aux.cpu().detach().numpy())

                    del pred_P_1, pred_P_2, pred_c, data_src

                # #### source seg & edge loss ####
                loss_src_seg_main = self.celoss(logits_list_src[0], labels_src)
                loss_src_seg_aux = 0.1 * self.celoss(logits_list_src[1],
                                                     labels_src)
                loss_src_seg = loss_src_seg_main + loss_src_seg_aux
                loss_src_seg.backward()
                # reprod_logger.add("loss_src_seg_{}".format(iter), loss_src_seg.cpu().detach().numpy())

                loss_dict["source_main"] = loss_src_seg_main.numpy()[0]
                loss_dict["source_aux"] = loss_src_seg_aux.numpy()[0]
                del loss_src_seg, images_src, loss_src_seg_aux, loss_src_seg_main

                #### aug loss #######
                # reprod_logger.add("images_tgt_{}".format(iter), images_tgt.cpu().detach().numpy())
                augs = augmentation.get_augmentation()
                images_tgt_aug, labels_tgt_aug = augmentation.augment(
                    images=images_tgt.cpu(),
                    labels=labels_tgt_psu.detach().cpu(),
                    aug=augs,
                    logger=reprod_logger,
                    iters="{}_1".format(iter))
                # reprod_logger.add("images_tgt_aug_{}".format(iter), images_tgt_aug.cpu().detach().numpy())
                # reprod_logger.add("labels_tgt_aug_{}".format(iter), labels_tgt_aug.cpu().detach().numpy())
                # reprod_logger.add("labels_tgt_aug_aux_{}".format(iter), labels_tgt_aug_aux.cpu().detach().numpy())
                images_tgt_aug = images_tgt_aug.cuda()
                labels_tgt_aug = labels_tgt_aug.cuda()

                _, labels_tgt_aug_aux = augmentation.augment(
                    images=images_tgt.cpu(),
                    labels=labels_tgt_psu_aux.detach().cpu(),
                    aug=augs,
                    logger=reprod_logger,
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
                loss_tgt_aug.backward()

                # res_tgt_aug = logits_list_tgt_aug[0] + logits_list_tgt_aug[1]
                # reprod_logger.add("res_tgt_aug_{}".format(iter), res_tgt_aug.cpu().detach().numpy())
                # reprod_logger.add("loss_tgt_aug_{}".format(iter), loss_tgt_aug.cpu().detach().numpy())

                loss_dict['target_aug_main'] = loss_tgt_aug_main.numpy()[0]
                loss_dict['target_aug_aux'] = loss_tgt_aug_aux.numpy()[0]
                del images_tgt_aug, labels_tgt_aug, labels_tgt_aug_aux, images_tgt, \
                    logits_list_tgt_aug, loss_tgt_aug, loss_tgt_aug_aux, loss_tgt_aug_main

                ####  target seg & edge loss ####
                if edgeconstrain:
                    edges_tgt = F.mask_to_binary_edge(
                        labels_tgt_psu,
                        radius=2,
                        num_classes=train_dataset_tgt.num_classes)
                    loss_src_edge = self.bceloss(logits_list_src[2], edges_src)

                    if iter > 60000:
                        loss_tgt_seg = self.celoss(logits_list_tgt[0], labels_tgt) \
                                        + 0.1 * self.celoss(logits_list_tgt[1], labels_tgt_psu_aux)
                        loss_tgt_edge = self.bceloss(logits_list_tgt[2],
                                                     edges_tgt)
                        loss_edge = loss_tgt_seg + loss_tgt_edge + loss_src_edge
                    else:
                        loss_tgt_seg = paddle.zeros([1])
                        loss_tgt_edge = paddle.zeros([1])
                        loss_edge = loss_src_edge

                    loss_edge.backward()

                    loss_dict['target_seg'] = loss_tgt_seg.numpy()[0]
                    loss_dict['target_edge'] = loss_tgt_edge.numpy()[0]
                    loss_dict['source_edge'] = loss_src_edge.numpy()[0]

                    del loss_src_edge, loss_tgt_edge, loss_src_edge

                #### edge input seg; src & tgt edge pull in ######
                if edgepullin:
                    if nranks > 1:
                        logits_list_edge_src = ddp_model(logits_list_src[2])
                        logits_list_edge_tgt = ddp_model(logits_list_tgt[2])
                    else:
                        logits_list_edge_src = self.model(logits_list_src[2])
                        logits_list_edge_tgt = self.model(logits_list_tgt[2])

                    loss_src_edge_seg = self.celoss(logits_list_edge_src[0], labels_src) \
                                    + 0.1 * self.celoss(logits_list_edge_src[1], labels_src)
                    loss_tgt_edge_seg = self.celoss(logits_list_edge_tgt[0], labels_tgt) \
                                    + 0.1 * self.celoss(logits_list_edge_tgt[1], labels_tgt_psu_aux)
                    loss_edge_pullin = self.klloss(logits_list_edge_tgt[2], (logits_list_edge_tgt[2]+logits_list_edge_src[2])/2) \
                                    + self.klloss(logits_list_edge_src[2], (logits_list_edge_tgt[2]+logits_list_edge_src[2])/2)

                    loss_src_edge_seg = loss_src_edge_seg + loss_tgt_edge_seg + loss_edge_pullin
                    loss_src_edge_seg.backward()
                    loss_dict['source_edge_seg'] = loss_src_edge_seg.numpy()[0]
                    loss_dict['target_edge_seg'] = loss_tgt_edge_seg.numpy()[0]
                    loss_dict['tgt_src_edge_pullin'] = loss_edge_pullin.numpy(
                    )[0]

                    del loss_src_edge_seg, loss_tgt_edge_seg, loss_edge_pullin

                #### mask input feature & pullin  ######
                if featurepullin:
                    print(1)
                    # labels_src_onehot = F.one_hot(labels_src, train_dataset_src.num_classes)

                loss = sum(loss_dict.values())
                # reprod_logger.add("loss_total_{}".format(iter), np.array([loss]))
                # for name, tensor in self.model.named_parameters():
                #     if not tensor.stop_gradient:
                #         grad = tensor.grad
                #         try:
                #             self.reprod_logger.add(name, np.array([grad]))
                #         except AttributeError:
                #             print(name, "does not have grad but stop gradients=", tensor.stop_gradient)

                optimizer.step()
                # self.ema.update_params()

                with paddle.no_grad():

                    ##### log & save #####
                    lr = optimizer.get_lr()
                    # reprod_logger.add("lr_{}".format(iter), np.array([lr]))

                    # update lr
                    if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                        lr_sche = optimizer.user_defined_optimizer._learning_rate
                    else:
                        lr_sche = optimizer._learning_rate
                    if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                        lr_sche.step()

                    self.model.clear_gradients()

                    batch_cost_averager.record(
                        time.time() - batch_start, num_samples=batch_size)

                    iter += 1
                    if (iter) % log_iters == 0 and local_rank == 0:
                        import functools
                        label_tgt_acc = ((labels_tgt == labels_tgt_psu).numpy().sum().astype('float32')\
                                            /functools.reduce(lambda a, b: a * b, labels_tgt_psu.shape))*100

                        remain_iters = iters - iter
                        avg_train_batch_cost = batch_cost_averager.get_average()
                        avg_train_reader_cost = reader_cost_averager.get_average(
                        )
                        eta = calculate_eta(remain_iters, avg_train_batch_cost)
                        logger.info(
                            "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, pix_acc: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
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
                        num_workers = 1 if num_workers > 0 else 0

                        if test_config is None:
                            test_config = {}
                        # self.ema.apply_shadow()
                        # self.ema.model.eval()

                        PA, _, MIoU, _ = val.evaluate(
                            self.model,
                            val_dataset_tgt,
                            num_workers=num_workers,
                            **test_config)

                        # self.ema.restore()
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
                            # self.ema.model.state_dict(),
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
                            if MIoU > best_mean_iou:
                                best_mean_iou = MIoU
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
                                log_writer.add_scalar('Evaluate/mIoU', MIoU,
                                                      iter)
                                log_writer.add_scalar('Evaluate/PA', PA, iter)
                    batch_start = time.time()
            #     if iter > 3:
            #         break
            # reprod_logger.save("/ssd2/tangshiyu/Code/pixmatch/models/train_paddle.npy")
            # break
            # self.ema.update_buffer()

        # # Calculate flops.
        # if local_rank == 0:

        #     def count_syncbn(m, x, y):
        #         x = x[0]
        #         nelements = x.numel()
        #         m.total_ops += int(2 * nelements)

        #     _, c, h, w = images_src.shape
        #     flops = paddle.flops(
        #         self.model, [1, c, h, w],
        #         custom_ops={paddle.nn.SyncBatchNorm: count_syncbn})

        # Sleep for half a second to let dataloader release resources.
        time.sleep(0.5)
        if use_vdl:
            log_writer.close()
