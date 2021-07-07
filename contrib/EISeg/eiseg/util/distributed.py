import paddle
import paddle.distributed as dist
import numpy as np

def get_rank():
    return dist.get_rank()


def synchronize():
    return dist.barrier()


def get_world_size():
    return dist.get_world_size()


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict
    with paddle.no_grad():
        keys = []
        losses = []

        for k in loss_dict.keys():
            keys.append(k)
            loss = dist.all_reduce(loss_dict[k].astype('float32')) / paddle.distributed.get_world_size()
            losses.append(loss)

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses
