import os
import sys
import random
import argparse
from easydict import EasyDict as edict
import time

import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
import paddleseg.transforms as T
from paddleseg.utils import logger, get_sys_env, logger
from albumentations import (
    Compose,
    ShiftScaleRotate,
    PadIfNeeded,
    RandomCrop,
    RGBShift,
    RandomBrightnessContrast,
    RandomRotate90,
    HorizontalFlip,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from model.model import (
    get_hrnet_model,
    DistMapsHRNetModel,
    get_deeplab_model,
    get_shufflenet_model,
)
from model.modeling.hrnet_ocr import HighResolutionNet
from model.loss import *
from data.points_sampler import MultiPointSampler
from data.mdiy import MyDataset
from util.config import cfgData
from util.util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="cfg",
        type=str,
        default="./train_config.yaml",
        help="The config file.",
    )
    return parser.parse_args()


def main():
    env_info = get_sys_env()
    info = ["{}: {}".format(k, v) for k, v in env_info.items()]
    info = "\n".join(
        ["", format("Environment Information", "-^48s")] + info + ["-" * 48]
    )
    logger.info(info)
    place = (
        "gpu"
        if env_info["Paddle compiled with cuda"] and env_info["GPUs used"]
        else "cpu"
    )
    paddle.set_device(place)
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    cfg = cfgData(parse_args().cfg)
    model_cfg = edict()
    model_cfg.input_normalization = {
        "mean": [0.5, 0.5, 0.5],
        "std": [1, 1, 1],
    }
    model_cfg.num_max_points = 10
    model_cfg.input_transform = T.Compose(
        [
            T.Normalize(
                mean=model_cfg.input_normalization["mean"],
                std=model_cfg.input_normalization["std"],
            )
        ],
        to_rgb=False,
    )
    nn.initializer.set_global_initializer(
        nn.initializer.Normal(), nn.initializer.Constant()
    )
    models = cfg.get("model")
    if models.get("type") == "deeplab":
        model = get_deeplab_model(
            backbone=models.get("backbone"), is_ritm=models.get("is_ritm")
        )
    elif models.get("type") == "hrnet":
        model = get_hrnet_model(
            width=models.get("width"),
            ocr_width=models.get("ocr_width"),
            with_aux_output=models.get("with_aux_output"),
            is_ritm=models.get("is_ritm"),
        )
    elif models.get("type") == "shufflenet":
        model = get_shufflenet_model()
    if models.get("weights") != "None":
        model.load_weights(models.get("weights"))
    backbone_params, other_params = model.get_trainable_params()

    if nranks > 1:
        if (
            not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized()
        ):
            paddle.distributed.init_parallel_env()
            ddp_net = paddle.DataParallel(model)
        else:
            ddp_net = paddle.DataParallel(model)
        train(
            ddp_net,
            cfg,
            model_cfg,
            backbone_params=backbone_params,
            other_params=other_params,
        )
    else:
        train(
            model,
            cfg,
            model_cfg,
            backbone_params=backbone_params,
            other_params=other_params,
        )


def train(model, cfg, model_cfg, backbone_params=None, other_params=None):
    local_rank = paddle.distributed.ParallelEnv().local_rank

    max_iters = cfg.get("iters")
    save_dir = cfg.get("save_dir")
    batch_size = cfg.get("batch_size") if cfg.get("batch_size") > 1 else 1
    val_batch_size = batch_size
    input_normalization = model_cfg.input_normalization
    crop_size = cfg.get("train_dataset").get("crop_size")
    log_iters = cfg.get("log_iters")
    save_interval = cfg.get("save_interval")
    num_masks = 1

    train_augmentator = Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.40)),
            HorizontalFlip(),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
            # RandomBrightnessContrast(
            #     brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75
            # ),
            # RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ],
        p=1.0,
    )
    val_augmentator = Compose(
        [
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
        ],
        p=1.0,
    )

    def scale_func(image_shape):
        return random.uniform(0.75, 1.25)

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.7,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )
    trainset = MyDataset(
        dataset_path=cfg.get("dataset").get("dataset_path"),
        folder_name=cfg.get("train_dataset").get("folder_name"),
        images_dir_name=cfg.get("dataset").get("image_name"),
        masks_dir_name=cfg.get("dataset").get("label_name"),
        num_masks=num_masks,
        augmentator=train_augmentator,
        points_from_one_object=False,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        keep_background_prob=0.0,
        image_rescale=scale_func,
        points_sampler=points_sampler,
        samples_scores_path=None,
        samples_scores_gamma=1.25,
    )
    valset = MyDataset(
        dataset_path=cfg.get("dataset").get("dataset_path"),
        folder_name=cfg.get("val_dataset").get("folder_name"),
        images_dir_name=cfg.get("dataset").get("image_name"),
        masks_dir_name=cfg.get("dataset").get("label_name"),
        augmentator=val_augmentator,
        num_masks=num_masks,
        points_from_one_object=False,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        image_rescale=scale_func,
        points_sampler=points_sampler,
    )
    batch_sampler = paddle.io.DistributedBatchSampler(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    loader = paddle.io.DataLoader(
        trainset,
        batch_sampler=batch_sampler,
        return_list=True,
    )
    val_batch_sampler = paddle.io.DistributedBatchSampler(
        valset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = paddle.io.DataLoader(
        valset,
        batch_sampler=val_batch_sampler,
        return_list=True,
    )

    if cfg.get("use_vdl"):
        from visualdl import LogWriter

        log_writer = LogWriter(save_dir)

    iters_per_epoch = len(batch_sampler)

    opt = None
    if cfg.get("optimizer").get("type") == "adam":
        opt = paddle.optimizer.Adam
    elif cfg.get("optimizer").get("type") == "sgd":
        opt = paddle.optimizer.SGD
    else:
        raise ValueError("Opt only have adam or sgd now.")
    lr = None
    if cfg.get("learning_rate").get("decay").get("type") == "poly":
        lr = paddle.optimizer.lr.PolynomialDecay
    else:
        raise ValueError("Lr only have poly now.")
    optimizer1 = opt(
        learning_rate=lr(
            float(cfg.get("learning_rate").get("value_1")),
            decay_steps=cfg.get("learning_rate").get("decay").get("steps"),
            end_lr=cfg.get("learning_rate").get("decay").get("end_lr"),
            power=cfg.get("learning_rate").get("decay").get("power"),
        ),
        parameters=other_params,
    )
    optimizer2 = opt(
        learning_rate=lr(
            float(cfg.get("learning_rate").get("value_1")),
            decay_steps=cfg.get("learning_rate").get("decay").get("steps"),
            end_lr=cfg.get("learning_rate").get("decay").get("end_lr"),
            power=cfg.get("learning_rate").get("decay").get("power"),
        ),
        parameters=backbone_params,
    )
    instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    model.train()
    # with open('mobilenet_model.txt', 'w') as f:
    #     for keys, values in model.state_dict().items():
    #         f.write(keys +'\t'+str(values.shape)+"\n")
    iters = 0
    avg_loss = 0.0
    while iters < max_iters:
        for data in loader:
            tic = time.time()
            iters += 1
            # print("begin ", iters)
            if iters > max_iters:
                break
            if len(data) == 3:
                images, points, masks = data
            else:
                images, points = data
                masks = None
            if masks is not None:
                batch_size, num_points, c, h, w = masks.shape
                masks = masks.reshape([batch_size * num_points, c, h, w])

            # print(points.numpy())
            # output = batch_forward(model, images, masks, points)
            #
            # img = images.numpy()[0]
            # img = np.moveaxis(img, 0, 2)
            # print(img.shape)
            # print(img.max(), img.min())
            # plt.imshow(((img + 0.5) * 255).astype("uint8"))
            # plt.show()
            # # cv2.imwrite("img.png", (img + 0.5) * 255)
            #
            # print(masks.numpy().shape)
            # mask = masks.numpy()[0]
            # mask = np.moveaxis(mask, 0, 2)
            # cv2.imwrite("mask.png", mask * 255)

            # output = model(images, points)
            # print('instance', output['instances'])
            # print('mask', masks)

            loss = instance_loss(output["instances"], masks)
            if "instances_aux" in output.keys():
                aux_loss = instance_aux_loss(output["instances_aux"], masks)
                total_loss = loss + 0.4 * aux_loss
            else:
                total_loss = loss
            avg_loss += total_loss.numpy()[0]
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()
            lr = optimizer1.get_lr()
            if isinstance(optimizer1._learning_rate, paddle.optimizer.lr.LRScheduler):
                optimizer1._learning_rate.step()
            if isinstance(optimizer2._learning_rate, paddle.optimizer.lr.LRScheduler):
                optimizer2._learning_rate.step()
            model.clear_gradients()
            if iters % log_iters == 0:
                avg_loss /= log_iters
                logger.info(
                    "Epoch={}, Step={}/{}, loss={:.4f}, lr={}".format(
                        (iters - 1) // iters_per_epoch + 1,
                        iters,
                        max_iters,
                        avg_loss,
                        lr,
                    )
                )
                if cfg.get("use_vdl"):
                    log_writer.add_scalar("Train/loss", avg_loss, iters)
                    log_writer.add_scalar("Train/lr", lr, iters)
                avg_loss = 0.0
            if (iters % save_interval == 0 or iters == max_iters) and local_rank == 0:
                model.eval()
                total_len = len(val_loader)
                val_iou = 0
                for val_num, val_data in enumerate(val_loader):
                    if len(data) == 3:
                        val_images, val_points, val_masks = val_data
                    else:
                        val_images, val_points = val_data
                        val_masks = None
                    if val_masks is not None:
                        (
                            val_batch_size,
                            val_num_points,
                            val_c,
                            val_h,
                            val_w,
                        ) = val_masks.shape
                        val_masks = val_masks.reshape(
                            [val_batch_size * val_num_points, val_c, val_h, val_w]
                        )
                    val_output = batch_forward(
                        model, val_images, val_masks, val_points, is_train=False
                    )["instances"]
                    # val_output = model(val_images, val_points)['instances']
                    # print('max', paddle.max(val_output))
                    # print('output shape', val_output.shape)
                    val_output = nn.functional.interpolate(
                        val_output,
                        mode="bilinear",
                        align_corners=True,
                        size=val_masks.shape[2:],
                    )
                    val_output = val_output > 0.5
                    iter_iou = get_iou(val_masks.numpy(), val_output.numpy())
                    val_iou += iter_iou
                logger.info(
                    "mean iou of iter {} is {}".format(iters, val_iou / total_len)
                )

                if cfg.get("use_vdl"):
                    log_writer.add_scalar("Eval/miou", val_iou / total_len, iters)

                current_save_dir = os.path.join(save_dir, "iter_{}".format(iters))

                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(
                    model.state_dict(), os.path.join(current_save_dir, "model.pdparams")
                )
                model.train()
            toc = time.time()
            print("ETA: ", toc - tic, (toc - tic) / 60 / 60 * (max_iters - iters))
            # print("end", iters)


def batch_forward(model, image, gt_mask, points, is_train=True):
    orig_image, orig_gt_mask = image.clone(), gt_mask.clone()
    prev_output = paddle.zeros_like(image, dtype="float32")[:, :1, ::]
    # last_click_indx = None
    num_iters = random.randint(1, 3)
    if is_train:
        model.eval()
    with paddle.no_grad():
        for click_indx in range(num_iters):
            net_input = paddle.concat([image, prev_output], axis=1)
            prev_output = model(net_input, points)["instances"]
            prev_output = nn.functional.sigmoid(prev_output)
            points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)
    if is_train:
        model.train()
    net_input = paddle.concat([image, prev_output], axis=1)
    output = model(net_input, points)
    return output


if __name__ == "__main__":
    main()
