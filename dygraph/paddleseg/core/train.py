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

import paddle
import paddle.nn.functional as F

from paddleseg.utils import Timer, calculate_eta, resume, logger
from paddleseg.core.val import evaluate


def check_logits_losses(logits, losses):
    len_logits = len(logits)
    len_losses = len(losses['types'])
    if len_logits != len_losses:
        raise RuntimeError(
            'The length of logits should equal to the types of loss config: {} != {}.'
            .format(len_logits, len_losses))


def loss_computation(logits, label, losses):
    check_logits_losses(logits, losses)
    loss = 0
    for i in range(len(logits)):
        logit = logits[i]
        #         if logit.shape[-2:] != label.shape[-2:]:
        #             logit = F.interpolate(
        #                 logit,
        #                 label.shape[-2:],
        #                 mode='bilinear',
        #                 align_corners=True,
        #                 align_mode=1)
        loss_i = losses['types'][i](logit, label)
        #         print(i, losses['coef'][i], loss_i)
        loss += losses['coef'][i] * loss_i


#     print('total loss:', loss)
    return loss


def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          use_vdl=False,
          losses=None):

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        # Initialize parallel training environment.
        paddle.distributed.init_parallel_env()
        strategy = paddle.distributed.prepare_context()
        ddp_model = paddle.DataParallel(model, strategy)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    if use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(save_dir)

    timer = Timer()
    avg_loss = 0.0
    iters_per_epoch = len(batch_sampler)
    best_mean_iou = -1.0
    best_model_iter = -1
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    timer.start()

    fp16 = False
    if fp16:
        print('turn on fp16!!!')
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break
            train_reader_cost += timer.elapsed_time()
            images = data[0]
            labels = data[1].astype('int64')

            #             img = images
            #             lab = labels
            #             print(img)
            #             print(lab)
            #             import numpy as np
            #             img = np.squeeze(img.numpy()).astype('uint8')
            #             print(img.shape)
            #             img = np.transpose(img, (1,2,0))
            #             lab = np.squeeze(lab.numpy()).astype('uint8')
            #             import cv2
            #             cv2.imwrite('img{}.png'.format(iter), img)
            #             cv2.imwrite('lab{}.png'.format(iter), lab)
            # #             exit()
            #             if iter==5:
            #                 exit()

            if fp16:
                images = paddle.reshape(images, images.shape)
                with paddle.amp.auto_cast():

                    if nranks > 1:
                        logits = ddp_model(images)
                    else:
                        logits = model(images)
                    loss = loss_computation(logits, labels, losses)

                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                lr = optimizer.get_lr()
                if isinstance(optimizer._learning_rate,
                              paddle.optimizer.lr.LRScheduler):
                    optimizer._learning_rate.step()
                model.clear_gradients()
            else:
                if nranks > 1:
                    logits = ddp_model(images)
                    loss = loss_computation(logits, labels, losses)
                    loss.backward()
                else:
                    #                     # debug finetune+mscale+loss
                    #                     print('iter: ', iter)
                    #                     import numpy as np
                    #                     images = np.load('/ssd1/home/chulutao/random-data/img.npy')
                    #                     labels = np.load(
                    #                         '/ssd1/home/chulutao/random-data/labels.npy')
                    #                     images = paddle.to_tensor(images)
                    #                     labels = paddle.to_tensor(labels)
                    #                     if iter == 1:
                    #                         print(images)
                    #                         print(labels)

                    #                     model.eval()

                    logits = model(images)
                    loss = loss_computation(logits, labels, losses)
                    loss.backward()
                optimizer.step()
                lr = optimizer.get_lr()
                if isinstance(optimizer._learning_rate,
                              paddle.optimizer.lr.LRScheduler):
                    optimizer._learning_rate.step()
                model.clear_gradients()


######################### debug

#                 # debug
#                 images = paddle.arange(6291456, dtype='float32')
#                 images = paddle.reshape(images, (1, 3, 1024, 2048))
#                 images = 5 * paddle.ones((1, 3, 1024, 2048), dtype='float32')
#                 labels = 9 * paddle.ones((1, 1, 1024, 2048), dtype='int64')
#                 images = paddle.reshape(paddle.arange(0, 30000).astype("float32"), [1,3,100,100])
#                 print(images)
#                 model.eval()
#                 logits = model(images)

#                 print(logits)
#                 exit()

#                 import numpy as np
#                 np.random.seed(6)
#                 a = paddle.to_tensor(np.random.rand(1, 19, 1024, 2048).astype("float32"))
#                 b = paddle.to_tensor(np.random.rand(1, 19, 1024, 2048).astype("float32"))
#                 print(a, '\n', b)
#                 c = paddle.to_tensor(np.random.rand(1, 19, 1024, 2048).astype("float32"))
#                 d = paddle.to_tensor(np.random.rand(1, 19, 1024, 2048).astype("float32"))

#                 logits = [a, b, c, d]
#                 print(a)
#                 import numpy as np
#                 np.random.seed(6)
#                 labels = paddle.to_tensor(np.random.randint(20, size=(1,1,100,100)))
#                 print(labels)
#                 labels = paddle.ones((1, 1, 10, 10), dtype='int64')
#                 print('==========')
#                 loss = loss_computation(logits, labels, losses)
#                 print(loss)

#                 exit()
###############################

            avg_loss += loss.numpy()[0]
            train_batch_cost += timer.elapsed_time()
            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                avg_train_reader_cost = train_reader_cost / log_iters
                avg_train_batch_cost = train_batch_cost / log_iters
                train_reader_cost = 0.0
                train_batch_cost = 0.0
                remain_iters = iters - iter
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.4f} | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost, eta))
                if use_vdl:
                    log_writer.add_scalar('Train/loss', avg_loss, iter)
                    log_writer.add_scalar('Train/lr', lr, iter)
                    log_writer.add_scalar('Train/batch_cost',
                                          avg_train_batch_cost, iter)
                    log_writer.add_scalar('Train/reader_cost',
                                          avg_train_reader_cost, iter)
                avg_loss = 0.0

            if (iter % save_interval == 0
                    or iter == iters) and (val_dataset is not None):
                num_workers = 1 if num_workers > 0 else 0
                mean_iou, acc = evaluate(
                    model, val_dataset, num_workers=num_workers)
                model.train()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))

                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_dir = os.path.join(save_dir, "best_model")
                        paddle.save(
                            model.state_dict(),
                            os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                        '[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'
                        .format(best_mean_iou, best_model_iter))

                    if use_vdl:
                        log_writer.add_scalar('Evaluate/mIoU', mean_iou, iter)
                        log_writer.add_scalar('Evaluate/Acc', acc, iter)
            timer.restart()

    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    if use_vdl:
        log_writer.close()
