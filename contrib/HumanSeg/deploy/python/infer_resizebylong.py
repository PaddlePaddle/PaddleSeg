# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
# ==============================================================================
"""实时人像分割Python预测部署"""

import os
import argparse
import numpy as np
import cv2

import paddle.fluid as fluid


def humanseg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """计算光流跟踪匹配点和光流图
    输入参数:
        pre_gray: 上一帧灰度图
        cur_gray: 当前帧灰度图
        prev_cfd: 上一帧光流图
        dl_weights: 融合权重图
        disflow: 光流数据结构
    返回值:
        is_track: 光流点跟踪二值图，即是否具有光流点匹配
        track_cfd: 光流跟踪图
    """
    check_thres = 8
    hgt, wdh = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    # 计算前向光流
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    # 计算后向光流
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    get_round = lambda data: (int)(data + 0.5) if data >= 0 else (int)(data -
                                                                       0.5)
    for row in range(hgt):
        for col in range(wdh):
            # 计算光流处理后对应点坐标
            # (row, col) -> (cur_x, cur_y)
            fxy_fw = flow_fw[row, col]
            dx_fw = get_round(fxy_fw[0])
            cur_x = dx_fw + col
            dy_fw = get_round(fxy_fw[1])
            cur_y = dy_fw + row
            if cur_x < 0 or cur_x >= wdh or cur_y < 0 or cur_y >= hgt:
                continue
            fxy_bw = flow_bw[cur_y, cur_x]
            dx_bw = get_round(fxy_bw[0])
            dy_bw = get_round(fxy_bw[1])
            # 光流移动小于阈值
            lmt = ((dy_fw + dy_bw) * (dy_fw + dy_bw) +
                   (dx_fw + dx_bw) * (dx_fw + dx_bw))
            if lmt >= check_thres:
                continue
            # 静止点降权
            if abs(dy_fw) <= 0 and abs(dx_fw) <= 0 and abs(dy_bw) <= 0 and abs(
                    dx_bw) <= 0:
                dl_weights[cur_y, cur_x] = 0.05
            is_track[cur_y, cur_x] = 1
            track_cfd[cur_y, cur_x] = prev_cfd[row, col]
    return track_cfd, is_track, dl_weights


def humanseg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回值:
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    cur_cfd = dl_cfd.copy()
    idxs = np.where(is_track > 0)
    for i in range(len(idxs)):
        x, y = idxs[0][i], idxs[1][i]
        dl_score = dl_cfd[y, x]
        track_score = track_cfd[y, x]
        if dl_score > 0.9 or dl_score < 0.1:
            if dl_weights[x, y] < 0.1:
                cur_cfd[x, y] = 0.3 * dl_score + 0.7 * track_score
            else:
                cur_cfd[x, y] = 0.4 * dl_score + 0.6 * track_score
        else:
            cur_cfd[x, y] = dl_weights[x, y] * dl_score + (
                1 - dl_weights[x, y]) * track_score
    return cur_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    """设置背景和前景阈值mask
    输入参数:
        img : 原始图像, np.uint8 类型.
        thresh_bg : 背景阈值百分比，低于该值置为0.
        thresh_fg : 前景阈值百分比，超过该值置为1.
    返回值:
        dst : 原始图像设置完前景背景阈值mask结果, np.float32 类型.
    """
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optflow_handle(cur_gray, scoremap, is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        scoremap : 当前帧分割结果
        is_init : 是否第一帧
    Returns:
        dst : 光流追踪图和预测结果融合图, 类型为 np.float32
    """
    width, height = scoremap.shape[0], scoremap.shape[1]
    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev_gray = np.zeros((height, width), np.uint8)
    prev_cfd = np.zeros((height, width), np.float32)
    cur_cfd = scoremap.copy()
    if is_init:
        is_init = False
        if height <= 64 or width <= 64:
            disflow.setFinestScale(1)
        elif height <= 160 or width <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((width, height), np.float32) * 0.3
        track_cfd, is_track, weights = humanseg_tracking(
            prev_gray, cur_gray, prev_cfd, weights, disflow)
        fusion_cfd = humanseg_track_fuse(track_cfd, cur_cfd, weights, is_track)
    fusion_cfd = cv2.GaussianBlur(fusion_cfd, (3, 3), 0)
    return fusion_cfd


class HumanSeg:
    """人像分割类
    封装了人像分割模型的加载，数据预处理，预测，后处理等
    """

    def __init__(self, model_dir, mean, scale, long_size, use_gpu=False):

        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.long_size = long_size
        self.load_model(model_dir, use_gpu)

    def load_model(self, model_dir, use_gpu):
        """加载模型并创建predictor
        Args:
            model_dir: 预测模型路径, 包含 `__model__` 和 `__params__`
            use_gpu: 是否使用GPU加速
        """
        prog_file = os.path.join(model_dir, '__model__')
        params_file = os.path.join(model_dir, '__params__')
        config = fluid.core.AnalysisConfig(prog_file, params_file)
        if use_gpu:
            config.enable_use_gpu(100, 0)
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
        config.disable_glog_info()
        config.switch_specify_input_names(True)
        config.enable_memory_optim()
        self.predictor = fluid.core.create_paddle_predictor(config)

    def preprocess(self, image):
        """图像预处理
        hwc_rgb 转换为 chw_bgr，并进行归一化
        输入参数:
            image: 原始图像
        返回值:
            经过预处理后的图片结果
        """
        origin_h, origin_w = image.shape[0], image.shape[1]
        scale = float(self.long_size) / max(origin_w, origin_h)
        resize_w = int(round(origin_w * scale))
        resize_h = int(round(origin_h * scale))
        img_mat = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        pad_h = self.long_size - resize_h
        pad_w = self.long_size - resize_w
        img_mat = cv2.copyMakeBorder(
            img_mat,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=[127.5, 127.5, 127.5])

        # HWC -> CHW
        img_mat = img_mat.swapaxes(1, 2)
        img_mat = img_mat.swapaxes(0, 1)
        # Convert to float
        img_mat = img_mat[:, :, :].astype('float32')
        img_mat = (img_mat / 255. - self.mean) / self.scale
        img_mat = img_mat[np.newaxis, :, :, :]
        return img_mat

    def postprocess(self, image, output_data):
        """对预测结果进行后处理
        Args:
             image: 原始图，opencv 图片对象
             output_data: Paddle预测结果原始数据
        Returns:
             原图和预测结果融合并做了光流优化的结果图
        """
        scoremap = output_data[0, 1, :, :]
        scoremap = (scoremap * 255).astype(np.uint8)
        # 光流处理
        cur_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        origin_h, origin_w = image.shape[0], image.shape[1]
        scale = float(self.long_size) / max(origin_w, origin_h)
        resize_w = int(round(origin_w * scale))
        resize_h = int(round(origin_h * scale))
        cur_gray = cv2.resize(
            cur_gray, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        pad_h = self.long_size - resize_h
        pad_w = self.long_size - resize_w
        cur_gray = cv2.copyMakeBorder(
            cur_gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=127.5)
        optflow_map = optflow_handle(cur_gray, scoremap, False)
        optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
        optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
        optflow_map = optflow_map[0:resize_h, 0:resize_w]
        optflow_map = cv2.resize(optflow_map, (origin_w, origin_h))
        optflow_map = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
        bg_im = np.ones_like(optflow_map) * 255
        comb = (optflow_map * image + (1 - optflow_map) * bg_im).astype(
            np.uint8)
        return comb

    def run_predict(self, image):
        """运行预测并返回可视化结果图
        输入参数:
            image: 需要预测的原始图, opencv图片对象
        返回值:
            可视化的预测结果图
        """
        im_mat = self.preprocess(image)
        im_tensor = fluid.core.PaddleTensor(im_mat.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[1]
        output_data = output_data.as_ndarray()
        return self.postprocess(image, output_data)


def predict_image(seg, image_path):
    """对图片文件进行分割
    结果保存到`result.jpeg`文件中
    """
    img_mat = cv2.imread(image_path)
    img_mat = seg.run_predict(img_mat)
    cv2.imwrite('result.jpeg', img_mat)


def predict_video(seg, video_path):
    """对视频文件进行分割
    结果保存到`result.avi`文件中
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 用于保存预测结果视频
    out = cv2.VideoWriter('result.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, height))
    # 开始获取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_mat = seg.run_predict(frame)
            out.write(img_mat)
        else:
            break
    cap.release()
    out.release()


def predict_camera(seg):
    """从摄像头获取视频流进行预测
    视频分割结果实时显示到可视化窗口中
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_mat = seg.run_predict(frame)
            cv2.imshow('HumanSegmentation', img_mat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def main(args):
    """预测程序入口
    完成模型加载, 对视频、摄像头、图片文件等预测过程
    """
    model_dir = args.model_dir
    use_gpu = args.use_gpu

    # 加载模型
    mean = [0.5, 0.5, 0.5]
    scale = [0.5, 0.5, 0.5]
    long_size = 192
    seg = HumanSeg(model_dir, mean, scale, long_size, use_gpu)
    if args.use_camera:
        # 开启摄像头
        predict_camera(seg)
    elif args.video_path:
        # 使用视频文件作为输入
        predict_video(seg, args.video_path)
    elif args.img_path:
        # 使用图片文件作为输入
        predict_image(seg, args.img_path)


def parse_args():
    """解析命令行参数
    """
    parser = argparse.ArgumentParser('Realtime Human Segmentation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='',
        help='path of human segmentation model')
    parser.add_argument(
        '--img_path', type=str, default='', help='path of input image')
    parser.add_argument(
        '--video_path', type=str, default='', help='path of input video')
    parser.add_argument(
        '--use_camera',
        type=bool,
        default=False,
        help='input video stream from camera')
    parser.add_argument(
        '--use_gpu', type=bool, default=False, help='enable gpu')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
