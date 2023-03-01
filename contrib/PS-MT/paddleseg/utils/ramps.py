import numpy as np


def gaussian_rampup(start, current, rampup_length):
    assert rampup_length >= 0
    if current == 0:
        return .0
    if current < start:
        return .0
    if current >= rampup_length:
        return 1.0
    return np.exp(-5 * (1 - current / rampup_length) ** 2)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length

    return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    return current / rampup_length


def cosine_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    # 截取函数，current中小于0的数变为0，大于ramip_length的变为ramip_length
    current = np.clip(current, 0.0, rampup_length)
    return 1 - float(.5 * (np.cos(np.pi * current / rampup_length) + 1))


def log_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(1 - np.exp(-5.0 * current / rampup_length))


def exp_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    return float(np.exp(5.0 * (current / rampup_length - 1)))
