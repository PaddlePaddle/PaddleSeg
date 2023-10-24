# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import paddle
import numpy as np
from paddleseg.cvlibs import Config
from paddleseg.cvlibs import SegBuilder
from paddleseg.utils import worker_init_fn, metrics
from paddleseg.core.infer import reverse_transform

from paddleslim.auto_compression import AutoCompression
from paddleslim.common import load_config as load_slim_config
from paddleslim.common.dataloader import get_feed_vars

eval_dataset = None


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of the config include data config.")
    parser.add_argument(
        '--act_config_path',
        type=str,
        default=None,
        help="path of the auto compression config.")
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    """
    计算模型在验证集上的评估指标。
    
    Args:
        exe: 执行器。
        compiled_test_program: 编译后的测试程序。
        test_feed_names: 测试输入的名称。
        test_fetch_list: 测试需要获取的变量列表。
    
    Returns:
        miou: IoU 平均值。
    """
    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True, )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    print("Start evaluating (total_samples: {}, total_iters: {})...".format(
        len(eval_dataset), total_iters))

    for iters, data in enumerate(loader):
        image, label = data['img'], data['label']
        label = np.array(label).astype('int64')
        image = np.array(image)

        logits = exe.run(compiled_test_program,
                         feed={test_feed_names[0]: image},
                         fetch_list=test_fetch_list,
                         return_numpy=True)

        paddle.disable_static()
        logit = logits[
            0]  # logit shape is 3, except  data['trans_info'] needs to be empty
        for i in range(len(data['trans_info'][::-1][0][1])):
            data['trans_info'][::-1][0][1][i] = paddle.to_tensor(data['trans_info'][::-1][0][1][i])
        logit = reverse_transform(
            paddle.to_tensor(logit).unsqueeze(0), data['trans_info'], mode='bilinear')
        pred = paddle.to_tensor(logit).squeeze(0)
        if len(
                pred.shape
        ) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            paddle.to_tensor(label),
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
        if iters % 100 == 0:
            print("Eval iter:", iters)

    _, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                               label_area_all)
    _, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    _, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)

    infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        len(eval_dataset), miou, acc, kappa, mdice)
    print(infor)

    paddle.enable_static()
    return miou


def reader_wrapper(reader, input_name):
    if isinstance(input_name, list) and len(input_name) == 1:
        input_name = input_name[0]

    def gen():
        for i, data in enumerate(reader()):
            imgs = np.array(data['img'])
            yield {input_name: imgs}

    return gen


def main(args):
    paddle.enable_static()
    rank_id = paddle.distributed.get_rank()

    if args.devices == 'gpu':
        place = paddle.CUDAPlace(rank_id)
        paddle.set_device('gpu')
    else:
        place = paddle.CPUPlace()
        paddle.set_device('cpu')

    # step1: load dataset config and create dataloader
    act_config = load_slim_config(args.act_config_path)
    assert os.path.exists(
        args.config_path), f"config path does't exist: {args.config_path}"

    data_cfg = Config(args.config_path)
    builder = SegBuilder(data_cfg)

    train_dataset = builder.train_dataset
    global eval_dataset
    eval_dataset = builder.val_dataset
    batch_size = data_cfg.batch_size

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(
        train_dataset,
        places=[place],
        batch_sampler=batch_sampler,
        num_workers=0,
        return_list=True,
        worker_init_fn=worker_init_fn)

    input_name = get_feed_vars(
        act_config['Global']['model_dir'],
        act_config['Global']['model_filename'], act_config['Global'][
            'params_filename'])  # get the name of forward input
    train_dataloader = reader_wrapper(train_loader, input_name)

    # step2: create and instance of AutoCompression
    ac = AutoCompression(
        model_dir=act_config['Global']['model_dir'],
        model_filename=act_config['Global']['model_filename'],
        params_filename=act_config['Global']['params_filename'],
        save_dir=args.save_dir,
        config=act_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function if rank_id == 0 else None,
        deploy_hardware=act_config['Global'].get('deploy_hardware') or None,
        input_shapes=act_config['Global'].get('input_shapes', None))

    # step3: start the compression job
    ac.compress()


if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    main(args)
