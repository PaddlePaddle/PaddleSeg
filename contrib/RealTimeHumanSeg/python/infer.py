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

import os
import sys

import numpy as np
import cv2

import paddle.fluid as fluid


def get_round(data):
    """
    get round of data
    """
    round = 0.5 if data >= 0 else -0.5
    return (int)(data + round)


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """
    human segmentation tracking
    """
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    for r in range(h):
        for c in range(w):
            fxy_fw = flow_fw[r, c]
            dx_fw = get_round(fxy_fw[0])
            cur_x = dx_fw + c
            dy_fw = get_round(fxy_fw[1])
            cur_y = dy_fw + r
            if cur_x < 0 or cur_x >= w or cur_y < 0 or cur_y >= h:
                continue
            fxy_bw = flow_bw[cur_y, cur_x]
            dx_bw = get_round(fxy_bw[0])
            dy_bw = get_round(fxy_bw[1])
            if ((dy_fw + dy_bw) * (dy_fw + dy_bw) + (dx_fw + dx_bw) * (dx_fw + dx_bw)) >= check_thres:
                continue
            if abs(dy_fw) <= 0 and abs(dx_fw) <= 0 and abs(dy_bw) <= 0 and abs(dx_bw) <= 0:
                dl_weights[cur_y, cur_x] = 0.05
            is_track[cur_y, cur_x] = 1
            track_cfd[cur_y, cur_x] = prev_cfd[r, c]

    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """
    human segmentation tracking fuse
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
            cur_cfd[x, y] = dl_weights[x,y]*dl_score + (1-dl_weights[x,y])*track_score
    return cur_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    """
    threshold mask
    """
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optflow_handle(cur_gray, scoremap, prev_gray, pre_cfd, disflow, is_init):
    """
    optical flow handling
    """
    w, h = scoremap.shape[0], scoremap.shape[1]
    cur_cfd = scoremap.copy()
    if is_init:
        is_init = False
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((w,h), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(prev_gray, cur_gray, pre_cfd,  weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)
    fusion_cfd = cv2.GaussianBlur(fusion_cfd, (3,3), 0)
    return fusion_cfd


class HumanSeg:
    """
    Human Segmentation Class
    """
    def __init__(self, model_dir, mean, scale, eval_size, use_gpu=False):
        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.eval_size = eval_size
        self.load_model(model_dir, use_gpu)

    def load_model(self, model_dir, use_gpu):
        """
        Load model from model_dir
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
        """
        preprocess image: hwc_rgb to chw_bgr
        """
        img_mat = cv2.resize(
            image, self.eval_size, interpolation=cv2.INTER_LINEAR)
        # HWC -> CHW
        img_mat = img_mat.swapaxes(1, 2)
        img_mat = img_mat.swapaxes(0, 1)
        # Convert to float
        img_mat = img_mat[:, :, :].astype('float32')
        # img_mat = (img_mat - mean) * scale
        img_mat = img_mat - self.mean
        img_mat = img_mat * self.scale
        img_mat = img_mat[np.newaxis, :, :, :]
        return img_mat

    def postprocess(self, image, output_data):
        """
        postprocess result: merge background with segmentation result
        """
        scoremap = output_data[0, 1, :, :]
        scoremap = (scoremap * 255).astype(np.uint8)
        ori_h, ori_w = image.shape[0], image.shape[1]
        evl_h, evl_w = self.eval_size[0], self.eval_size[1]
        disflow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        prev_gray = np.zeros((evl_h, evl_w), np.uint8)
        prev_cfd = np.zeros((evl_h, evl_w), np.float32)
        cur_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray, (evl_w, evl_h))
        optflow_map = optflow_handle(cur_gray, scoremap, prev_gray, prev_cfd, disflow, False)
        optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
        optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
        optflow_map = cv2.resize(optflow_map, (ori_w, ori_h))
        optflow_map = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
        bg = np.ones_like(optflow_map) * 255
        comb = (optflow_map * image + (1 - optflow_map) * bg).astype(np.uint8)
        return comb

    def run_predict(self, image):
        """
        run predict: return segmentation image mat
        """
        ori_im = image.copy()
        im_mat = self.preprocess(ori_im)
        im_tensor = fluid.core.PaddleTensor(im_mat.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[0]
        output_data = output_data.as_ndarray()
        return self.postprocess(image, output_data)


def predict_image(seg, image_path):
    """
    Do Predicting on a single image
    """
    img_mat = cv2.imread(image_path)
    img_mat = seg.run_predict(img_mat)
    cv2.imwrite('result.jpeg', img_mat)


def predict_video(seg, video_path):
    """
    Do Predicting on a video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Result Video Writer
    out = cv2.VideoWriter('result.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, height))
    id = 1
    # Start capturing from video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_mat = seg.run_predict(frame)
            out.write(img_mat)
            id += 1
            if id >= 51:
                break
        else:
            break
    cap.release()
    out.release()


def predict_camera(seg):
    """
    Do Predicting on a camera video stream: Press q to exit
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


def main(argv):
    """
    Entrypoint of the script
    """
    if len(argv) < 3:
        print('Usage: python infer.py /path/to/model/ /path/to/video')
        return

    model_dir = sys.argv[1]
    input_path = sys.argv[2]
    use_gpu = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    # Init model
    mean = [104.008, 116.669, 122.675]
    scale = [1.0, 1.0, 1.0]
    eval_size = (192, 192)
    seg = HumanSeg(model_dir, mean, scale, eval_size, use_gpu)
    # Run Predicting on a video and result will be saved as result.avi
    predict_camera(seg)
    #predict_video(seg, input_path)
    #predict_image(seg, input_path)


if __name__ == "__main__":
    main(sys.argv)
