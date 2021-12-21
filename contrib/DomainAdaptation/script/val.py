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
import tqdm
import numpy as np
from PIL import Image
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
import datasets
from utils import save_imgs

np.set_printoptions(suppress=True)

np.seterr(divide='ignore', invalid='ignore')

# Names
name_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafflight',
    'traffsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled'
]

synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class Eval():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)
        self.ignore_index = None
        self.synthia = True if num_class == 16 else False

    def pixel_accuracy(self):
        if np.sum(self.confusion_matrix) == 0:
            print("Attention: pixel_total is zero!!!")
            PA = 0
        else:
            PA = np.diag(
                self.confusion_matrix).sum() / self.confusion_matrix.sum()

        return PA

    def mean_pixel_accuracy(self, out_16_13=False):
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.synthia:
            MPA_16 = np.nanmean(MPA[:self.ignore_index])
            MPA_13 = np.nanmean(MPA[synthia_set_16_to_13])
            return MPA_16, MPA_13
        if out_16_13:
            MPA_16 = np.nanmean(MPA[synthia_set_16])
            MPA_13 = np.nanmean(MPA[synthia_set_13])
            return MPA_16, MPA_13
        MPA = np.nanmean(MPA[:self.ignore_index])

        return MPA

    def mean_iou(self, out_16_13=False):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(
                self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        if self.synthia:
            MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
            MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
            return MIoU_16, MIoU_13
        if out_16_13:
            MIoU_16 = np.nanmean(MIoU[synthia_set_16])
            MIoU_13 = np.nanmean(MIoU[synthia_set_13])
            return MIoU_16, MIoU_13
        MIoU = np.nanmean(MIoU[:self.ignore_index])

        return MIoU

    def fwiou(self, out_16_13=False):
        FWIoU = np.multiply(
            np.sum(self.confusion_matrix, axis=1),
            np.diag(self.confusion_matrix))
        FWIoU = FWIoU / (np.sum(self.confusion_matrix, axis=1) + np.sum(
            self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        if self.synthia:
            FWIoU_16 = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(
                self.confusion_matrix)
            FWIoU_13 = np.sum(i for i in FWIoU[synthia_set_16_to_13]
                              if not np.isnan(i)) / np.sum(
                                  self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        if out_16_13:
            FWIoU_16 = np.sum(
                i for i in FWIoU[synthia_set_16] if not np.isnan(i)) / np.sum(
                    self.confusion_matrix)
            FWIoU_13 = np.sum(
                i for i in FWIoU[synthia_set_13] if not np.isnan(i)) / np.sum(
                    self.confusion_matrix)
            return FWIoU_16, FWIoU_13
        FWIoU = np.sum(i for i in FWIoU if not np.isnan(i)) / np.sum(
            self.confusion_matrix)

        return FWIoU

    def mean_precision(self, out_16_13=False):
        Precision = np.diag(
            self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        if self.synthia:
            Precision_16 = np.nanmean(Precision[:self.ignore_index])
            Precision_13 = np.nanmean(Precision[synthia_set_16_to_13])
            return Precision_16, Precision_13
        if out_16_13:
            Precision_16 = np.nanmean(Precision[synthia_set_16])
            Precision_13 = np.nanmean(Precision[synthia_set_13])
            return Precision_16, Precision_13
        Precision = np.nanmean(Precision[:self.ignore_index])
        return Precision

    def print_every_class_eval(self, out_16_13=False, logger=None):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(
                self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        MPA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Precision = np.diag(
            self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        Class_ratio = np.sum(
            self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        Pred_retio = np.sum(
            self.confusion_matrix, axis=0) / np.sum(self.confusion_matrix)
        log_fn = print if logger is None else logger.info
        log_fn('===>Everyclass:\t' + 'MPA\t' + 'MIoU\t' + 'PC\t' + 'Ratio\t' +
               'Pred_Retio')
        # if out_16_13: MIoU = MIoU[synthia_set_16]
        for ind_class in range(len(MIoU)):
            pa = str(round(MPA[ind_class] * 100,
                           2)) if not np.isnan(MPA[ind_class]) else 'nan'
            iou = str(round(MIoU[ind_class] * 100,
                            2)) if not np.isnan(MIoU[ind_class]) else 'nan'
            pc = str(round(Precision[ind_class] * 100,
                           2)) if not np.isnan(Precision[ind_class]) else 'nan'
            cr = str(
                round(Class_ratio[ind_class] * 100,
                      2)) if not np.isnan(Class_ratio[ind_class]) else 'nan'
            pr = str(round(Pred_retio[ind_class] * 100,
                           2)) if not np.isnan(Pred_retio[ind_class]) else 'nan'
            log_fn('===>' + name_classes[ind_class] + ':\t' + pa + '\t' + iou +
                   '\t' + pc + '\t' + cr + '\t' + pr)

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)


def evaluate(model,
             eval_dataset,
             num_workers=0,
             print_detail=True,
             save_img=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    logger.info('Validating')
    evaluator = Eval(eval_dataset.NUM_CLASSES)
    evaluator.reset()

    model.eval()

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()

    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=True)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    progbar_val = progbar.Progbar(
        target=len(loader), verbose=0 if nranks < 2 else 2)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    with paddle.no_grad():
        for idx, (x, y, _, item) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)

            # Forward
            y = y.astype('int64')
            pred = model(x)  # 1, c, h, w
            if len(pred) > 1:
                pred = pred[0]

            # Convert to numpy
            label = y.squeeze(axis=1).numpy()  #
            argpred = np.argmax(pred.numpy(), axis=1)  # 1, 1, H, W
            if save_img:
                save_imgs(argpred, item, './output/')

            # Add to evaluator
            evaluator.add_batch(label, argpred)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail and idx % 10 == 0:
                progbar_val.update(idx + 1, [('batch_cost', batch_cost),
                                             ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

        PA = evaluator.pixel_accuracy()
        MPA = evaluator.mean_pixel_accuracy()
        MIoU = evaluator.mean_iou()
        FWIoU = evaluator.fwiou()
        PC = evaluator.mean_precision()
        logger.info(
            'PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.
            format(PA, MPA, MIoU, FWIoU, PC))

    return PA, MPA, MIoU, FWIoU
