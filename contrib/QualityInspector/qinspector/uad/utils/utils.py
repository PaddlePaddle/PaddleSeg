# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from skimage import measure, morphology
from skimage.segmentation import mark_boundaries

import paddle


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cholesky_inverse(input, upper=False, out=None, inplace=True):
    u = paddle.cholesky(input, upper)
    u = paddle.linalg.triangular_solve(u, paddle.eye(u.shape[-1]), upper=upper)
    if len(u.shape) == 2:
        uit = u.T
    elif len(u.shape) == 3:
        uit = paddle.transpose(u, perm=(0, 2, 1))
    elif len(u.shape) == 4:
        uit = paddle.transpose(u, perm=(0, 1, 3, 2))
    if inplace:
        input = u @uit if upper else uit @u
        return input
    else:
        out = u @uit if upper else uit @u
        return out


def mahalanobis(embedding, mean, inv_covariance):
    B, C, H, W = embedding.shape
    delta = (embedding - mean).reshape((B, C, H * W)).transpose((2, 0, 1))
    distances = ((delta @inv_covariance) @delta).sum(2).transpose((1, 0))
    distances = distances.reshape((B, H, W))
    distances = distances.sqrt_()
    return distances


def mahalanobis_einsum(embedding, mean, inv_covariance):
    M = embedding - mean
    distances = paddle.einsum('nmhw,hwmk,nkhw->nhw', M, inv_covariance, M)
    distances = distances.sqrt_()
    return distances


def svd_orthogonal(fin, fout, use_paddle=False):
    assert fin > fout, 'fin > fout'
    if use_paddle:
        X = paddle.rand((fout, fin))
        U, _, Vt = paddle.linalg.svd(X, full_matrices=False)
        # print(Vt.shape)
        # print(paddle.allclose(Vt@Vt.T, paddle.eye(Vt.shape[0])))
    else:
        X = np.random.random((fout, fin))
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        # print(Vt.shape)
        # print(np.allclose((Vt@ Vt.T), np.eye(Vt.shape[0])))
    W = paddle.to_tensor(Vt, dtype=paddle.float32).T
    return W


def orthogonal(rows, cols, gain=1):
    r"""return a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013).
    Args:
        rows: rows
        cols: cols
        gain: optional scaling factor
    Examples:
        >>> orthogonal_(5, 3)
    """
    flattened = paddle.randn((rows, cols))
    if rows < cols: flattened = flattened.T

    # Compute the qr factorization
    q, r = paddle.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = paddle.diag(r, 0)
    q *= d.sign()

    if rows < cols: q = q.T
    q *= gain
    return q


def cdist(X, Y, p=2.0):
    # 2d P, C = X.shape| R, C = Y.shape -> P,R
    P, C = X.shape
    R, C = Y.shape
    # 3d B, P, C = X.shape|1, R, C = Y.shape -> B, P,R
    # D = paddle.linalg.norm(X[:, None, :]-Y[None, :, :], axis=-1)
    """D = paddle.zeros((P, R))
    for i in range(P):
        D[i,:] = paddle.linalg.norm(X[i, None, :]-Y, axis=-1)
        #D[i,:] = (X[i, None, :]-Y).square().sum(-1).sqrt_()
    #"""
    D = []
    for i in range(P):
        D.append(paddle.linalg.norm(X[i, None, :] - Y, axis=-1))
    D = paddle.stack(D, 0)
    return D


def compute_pro_(y_true: np.ndarray, binary_amaps: np.ndarray,
                 method='mean') -> float:
    pros = []
    for binary_amap, mask in zip(binary_amaps, y_true):
        per_region_tpr = []
        for region in measure.regionprops(measure.label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
            per_region_tpr.append(TP_pixels / region.area)
        if method == 'mean' and per_region_tpr:
            pros.append(np.mean(per_region_tpr))
        else:
            pros.extend(per_region_tpr)
    return np.mean(pros)


def kthvalue(x: np.ndarray, k: int):
    return x[x.argpartition(k)[k]]


def get_thresholds(t: np.ndarray, num_samples=1000, reverse=False, opt=True):
    if opt:
        # use the worst-case for efficient determination of thresholds
        max_idx = t.reshape(t.shape[0], -1).max(1).argmax(0)
        t = t[max_idx].flatten()
        # return [kthvalue(t, max(1, math.floor(t.size * i / num_samples)-1)-1)
        #            for i in range(num_samples, 0, -1)]
        r = np.linspace(0, t.size - 1, num=num_samples).astype(int)
        if reverse: r = r[::-1]
        t.sort()
        return t[r]
        # idx = np.argsort(t)
        # return [t[idx[max(1, math.floor(t.size * i / num_samples)-1)-1]] for i in range(num_samples, 0, -1)]
    else:
        # return [kthvalue(t.flatten(), max(1, math.floor(t.size * i / num_samples)))
        #            for i in range(num_samples, 0, -1)]

        r = np.linspace(t.min(), t.max(), num=num_samples)
        if reverse: r = r[::-1]
        return r


def compute_pro(y_true: np.ndarray, amaps: np.ndarray, steps=500) -> float:
    y_true = y_true.squeeze()

    pros = []
    fprs = []
    for th in tqdm(get_thresholds(amaps, steps, True,
                                  True)):  # thresholds[::-1]:#
        binary_amaps = amaps.squeeze() > th
        """
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pro.append(TP_pixels / region.area)
        pros.append(np.mean(pro))"""
        pros.append(compute_pro_(y_true, binary_amaps, 'mean'))

        inverse_masks = 1 - y_true
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fprs.append(fpr)
        if fpr > 0.3: break

    # print(np.array(list(zip(pros,fprs))))
    fprs = np.array(fprs)
    pros = np.array(pros)
    return fprs, pros


# ported from OrthoAD repo
def compute_non_partial_auc(fpr, pro, at_fpr=1.0):
    acut = 0.  # area cut
    area = 0.  # area all
    assert 1 < len(pro)
    assert len(fpr) == len(pro)
    for i in range(len(fpr)):
        # calculate bin_size
        if len(fpr) - 1 != i:
            fpr_right = fpr[i + 1]
        else:
            fpr_right = 1.0
        b_left = (fpr[i] - fpr[i - 1]) / 2
        b_right = (fpr_right - fpr[i]) / 2
        if 0 == i:  # left-end
            b = fpr[i] + b_right
        elif len(fpr) - 1 == i:  # right-end
            b = b_left + 1. - fpr[i]
        else:
            b = b_left + b_right
        # calculate area
        if fpr[i] + b_right > at_fpr:
            b_cut = max(0, at_fpr - fpr[i] + b_left)  # bin cut
            acut += b_cut * pro[i]
        else:
            acut += b * pro[i]
        area += b * pro[i]
    return acut / at_fpr


def compute_roc(y_true: np.ndarray, amaps: np.ndarray, steps=500) -> float:
    y_true = y_true.squeeze()

    tprs = []
    fprs = []
    for th in tqdm(get_thresholds(amaps, steps, True,
                                  True)):  # thresholds[::-1]:#
        binary_amaps = amaps.squeeze() > th
        TP_pixels = np.logical_and(y_true, binary_amaps).sum()
        tpr = TP_pixels / y_true.sum()
        tprs.append(tpr)

        inverse_masks = 1 - y_true
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fprs.append(fpr)

    # print(np.array(list(zip(pros,fprs))))
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    return fprs, tprs


def compute_pro_score(y_true: np.ndarray,
                      amaps: np.ndarray,
                      steps=500,
                      non_partial_AUC=False) -> float:
    fprs, pros = compute_pro(y_true, amaps, steps)
    return compute_non_partial_auc(
        rescale(fprs), rescale(pros)) if non_partial_AUC else auc(
            rescale(fprs),
            rescale(pros))  # compute_non_partial_auc(fprs, rescale(pros), 0.3)


def compute_roc_score(y_true: np.ndarray,
                      amaps: np.ndarray,
                      steps=500,
                      non_partial_AUC=False) -> float:
    # fprs, tprs = compute_roc(masks, amaps, steps)
    fprs, tprs, thresholds = roc_curve(y_true, amaps)
    return compute_non_partial_auc(fprs, tprs) if non_partial_AUC else auc(fprs,
                                                                           tprs)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def plot_fig(test_img,
             scores,
             gts,
             threshold,
             save_dir,
             class_name,
             save_pic=True,
             tag=""):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    with_gt = gts != None
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4 + with_gt, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        if with_gt:
            gt = gts[i].transpose(1, 2, 0).squeeze()
            ax_img[1].imshow(gt, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[with_gt + 1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[with_gt + 1].imshow(img, cmap='gray', interpolation='none')
        ax_img[with_gt + 1].imshow(
            heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[with_gt + 1].title.set_text('Predicted heat map')
        ax_img[with_gt + 2].imshow(mask, cmap='gray')
        ax_img[with_gt + 2].title.set_text('Predicted mask')
        ax_img[with_gt + 3].imshow(vis_img)
        ax_img[with_gt + 3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        if i < 1:  # save one result
            if save_pic:
                save_name = os.path.join(save_dir,
                                         '{}_{}'.format(class_name, tag
                                                        if tag else i))
                fig_img.savefig(save_name, dpi=100)
            else:
                plt.show()
        plt.close()
        return


def eval_metric(labels, scores, metric='roc'):
    if metric == 'pro':
        return pro(labels, scores)
    if metric == 'roc':
        return roc(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")


def roc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def pro(masks, scores):
    '''
        https://github.com/YoungGod/DFR/blob/a942f344570db91bc7feefc6da31825cf15ba3f9/DFR-source/anoseg_dfr.py#L447
    '''
    # per region overlap
    max_step = 4000
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr
        masks_neg = ~masks
        fpr = np.logical_and(masks_neg,
                             binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    expect_fpr = 0.3
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # # rescale fpr [0, 0.3] -> [0, 1]
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])  # need scale
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score
