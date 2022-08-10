# The video propagation and fusion code was heavily based on https://github.com/hkchengrex/MiVOS
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/hkchengrex/MiVOS/blob/main/LICENSE

import glob
import os

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

from eiseg.util.vis import get_palette


def load_video(path, min_side=480):
    frame_list = []
    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if min_side:
            h, w = frame.shape[:2]
            new_w = (w * min_side // min(w, h))
            new_h = (h * min_side // min(w, h))
            frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        frame_list.append(frame)
    frames = np.stack(frame_list, axis=0)
    return frames


def load_masks(path, min_side=None):
    fnames = sorted(glob.glob(os.path.join(path, '*.png')))
    frame_list = []

    first_frame = np.array(Image.open(fnames[0]))
    binary_mask = (first_frame.max() == 255)

    for i, fname in enumerate(fnames):
        if min_side:
            image = Image.open(fname)
            w, h = image.size
            new_w = (w * min_side // min(w, h))
            new_h = (h * min_side // min(w, h))
            frame_list.append(
                np.array(
                    image.resize((new_w, new_h), Image.NEAREST),
                    dtype=np.uint8))
        else:
            frame_list.append(np.array(Image.open(fname), dtype=np.uint8))

    frames = np.stack(frame_list, axis=0)
    if binary_mask:
        frames = (frames > 128).astype(np.uint8)
    return frames


def overlay_davis(image, mask, alpha=0.5, palette=None):
    """ Overlay segmentation on top of RGB image. from davis official"""
    result = image.copy()
    if mask is not None:
        if not palette:
            palette = get_palette(np.max(mask) + 1)
        palette = np.array(palette)
        rgb_mask = palette[mask.astype(np.uint8)]
        mask_region = (mask > 0).astype(np.uint8)
        result = (result * (1 - mask_region[:, :, np.newaxis]) + (1 - alpha) *
                  mask_region[:, :, np.newaxis] * result + alpha * rgb_mask)
        result = result.astype(np.uint8)
    return result


def aggregate_wbg(prob, keep_bg=False, hard=False):
    k, _, h, w = prob.shape
    new_prob = paddle.concat(
        [paddle.prod(
            1 - prob, axis=0, keepdim=True), prob], 0).clip(1e-7, 1 - 1e-7)
    logits = paddle.log((new_prob / (1 - new_prob)))

    if hard:
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, axis=0)
    else:
        return F.softmax(logits, axis=0)[1:]
