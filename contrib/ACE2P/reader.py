# -*- coding: utf-8 -*- 
import numpy as np
import paddle.fluid as fluid
from ACE2P.config import cfg
import cv2

def get_affine_points(src_shape, dst_shape, rot_grad=0):
    # 获取图像和仿射后图像的三组对应点坐标
    # 三组点为仿射变换后图像的中心点, [w/2,0], [0,0]，及对应原始图像的点
    if dst_shape[0] == 0 or dst_shape[1] == 0:
        raise Exception('scale shape should not be 0')

    # 旋转角度
    rotation = rot_grad * np.pi / 180.0
    sin_v = np.sin(rotation)
    cos_v = np.cos(rotation)

    dst_ratio = float(dst_shape[0]) / dst_shape[1]
    h, w = src_shape
    src_ratio = float(h) / w if w != 0 else 0
    affine_shape = [h, h * dst_ratio] if src_ratio > dst_ratio \
                    else [w / dst_ratio, w]

    # 原始图像三组点
    points = [[0, 0]] * 3
    points[0] = (np.array([w, h]) - 1) * 0.5 
    points[1] = points[0] + 0.5 * affine_shape[0] * np.array([sin_v, -cos_v])
    points[2] = points[1] - 0.5 * affine_shape[1] * np.array([cos_v, sin_v])

    # 仿射变换后图三组点
    points_trans = [[0, 0]] * 3
    points_trans[0] = (np.array(dst_shape[::-1]) - 1) * 0.5
    points_trans[1] = [points_trans[0][0], 0]

    return points, points_trans

def preprocess(im):
    # ACE2P模型数据预处理
    im_shape = im.shape[:2]
    input_images = []
    for i, scale in enumerate(cfg.multi_scales):
        # 获取图像和仿射变换后图像的对应点坐标
        points, points_trans = get_affine_points(im_shape, scale)
        # 根据对应点集获得仿射矩阵
        trans = cv2.getAffineTransform(np.float32(points),
                                       np.float32(points_trans))
        # 根据仿射矩阵对图像进行仿射
        input = cv2.warpAffine(im,
                               trans,
                               scale[::-1],
                               flags=cv2.INTER_LINEAR)

        # 减均值测，除以方差，转换数据格式为NCHW
        input = input.astype(np.float32)
        input = (input / 255. - np.array(cfg.MEAN)) / np.array(cfg.STD)
        input = input.transpose(2, 0, 1).astype(np.float32)
        input = np.expand_dims(input, 0)

        # 水平翻转
        if cfg.flip:
            flip_input = input[:, :, :, ::-1]
            input_images.append(np.vstack((input, flip_input)))
        else:
            input_images.append(input)

    return input_images


def multi_scale_test(exe, test_prog, feed_name, fetch_list,
                        input_ims, im_shape):
    
    # 由于部分类别分左右部位, flipped_idx为其水平翻转后对应的标签
    flipped_idx = (15, 14, 17, 16, 19, 18)
    ms_outputs = []
    
    # 多尺度预测
    for idx, scale in enumerate(cfg.multi_scales):
        input_im = input_ims[idx]
        parsing_output = exe.run(program=test_prog,
                                 feed={feed_name[0]: input_im},
                                 fetch_list=fetch_list)
        output = parsing_output[0][0]
        if cfg.flip:
            # 若水平翻转，对部分类别进行翻转，与原始预测结果取均值
            flipped_output = parsing_output[0][1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            flipped_output = flipped_output[:, :, ::-1]
            output += flipped_output
            output *= 0.5

        output = np.transpose(output, [1, 2, 0])
        # 仿射变换回图像原始尺寸
        points, points_trans = get_affine_points(im_shape, scale)
        M = cv2.getAffineTransform(np.float32(points_trans), np.float32(points))
        logits_result = cv2.warpAffine(output, M, im_shape[::-1], flags=cv2.INTER_LINEAR)
        ms_outputs.append(logits_result)

    # 多尺度预测结果求均值，求预测概率最大的类别
    ms_fused_parsing_output = np.stack(ms_outputs)
    ms_fused_parsing_output = np.mean(ms_fused_parsing_output, axis=0)
    parsing = np.argmax(ms_fused_parsing_output, axis=2)
    return parsing, ms_fused_parsing_output

