import os.path as osp


def get_pretrain_weights(flag, backbone, save_dir):
    if flag is None:
        return None
    elif osp.isdir(flag):
        return flag
    else:
        raise Exception(
            "pretrain_weights need to be defined as directory path.")
