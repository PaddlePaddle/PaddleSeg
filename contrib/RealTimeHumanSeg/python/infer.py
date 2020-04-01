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
"""Python Inference solution for realtime humansegmentation"""

import os
import argparse
import numpy as np
import cv2

import paddle.fluid as fluid


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """Optical flow tracking for human segmentation
    Args:
        pre_gray: Grayscale of previous frame.
        cur_gray: Grayscale of current frame.
        prev_cfd: Optical flow of previous frame.
        dl_weights: Merged weights data.
        disflow: A data structure represents optical flow.
    Returns:
        is_track: Binary graph, whethe a pixel matched with a optical flow point.
        track_cfd: tracking optical flow image.
    """
    check_thres = 8
    hgt, wdh = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    # compute forward optical flow
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    # compute backword optical flow
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    get_round = lambda data: (int)(data + 0.5) if data >= 0 else (int)(data -0.5)
    for row in range(hgt):
        for col in range(wdh):
            # Calculate new coordinate after optfow process.
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
            # Filt the Optical flow point with a threshold
            lmt = ((dy_fw + dy_bw) * (dy_fw + dy_bw) + (dx_fw + dx_bw) * (dx_fw + dx_bw))
            if lmt >= check_thres:
                continue
            # Downgrade still points
            if abs(dy_fw) <= 0 and abs(dx_fw) <= 0 and abs(dy_bw) <= 0 and abs(dx_bw) <= 0:
                dl_weights[cur_y, cur_x] = 0.05
            is_track[cur_y, cur_x] = 1
            track_cfd[cur_y, cur_x] = prev_cfd[row, col]
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """Fusion of Optical flow track and segmentation
    Args:
        track_cfd: Optical flow track.
        dl_cfd: Segmentation result of current frame.
        dl_weights: Merged weights data.
        is_track: Binary graph, whethe a pixel matched with a optical flow point.
    Returns:
        cur_cfd: Fusion of Optical flow track and segmentation result.
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
            cur_cfd[x, y] = dl_weights[x, y] * dl_score + (1 - dl_weights[x, y]) * track_score
    return cur_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    """Threshold mask for image foreground and background
    Args:
        img : Original image, an instance of np.uint8 array.
        thresh_bg : Threshold for background, set to 0 when less than it.
        thresh_fg : Threshold for foreground, set to 1 when greater than it.
    Returns:
        dst : Image after set thresthold mask, ans instance of np.float32 array.
    """
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optflow_handle(cur_gray, scoremap, is_init):
    """Processing optical flow and segmentation result.
    Args:
        cur_gray : Grayscale of current frame.
        scoremap : Segmentation result of current frame.
        is_init : True only when process the first frame of a video.
    Returns:
        dst : Image after set thresthold mask, ans instance of np.float32 array.
    """
    width, height = scoremap.shape[0], scoremap.shape[1]
    disflow = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
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
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, prev_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)
    fusion_cfd = cv2.GaussianBlur(fusion_cfd, (3, 3), 0)
    return fusion_cfd


class HumanSeg:
    """Human Segmentation Class
    This Class instance will load the inference model and do inference
    on input image object.

    It includes the key stages for a object segmentation inference task.
    Call run_predict on your image and it will return a processed image.
    """
    def __init__(self, model_dir, mean, scale, eval_size, use_gpu=False):

        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.eval_size = eval_size
        self.load_model(model_dir, use_gpu)

    def load_model(self, model_dir, use_gpu):
        """Load paddle inference model.
        Args:
            model_dir: The inference model path includes `__model__` and `__params__`.
            use_gpu: Enable gpu if use_gpu is True
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
        """Preprocess input image.
        Convert hwc_rgb to chw_bgr.
        Args:
            image: The input opencv image object.
        Returns:
            A preprocessed image object.
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
        """Postprocess the inference result and original input image.
        Args:
             image: The original opencv image object.
             output_data: The inference output of paddle's humansegmentation model.
        Returns:
             The result merged original image and segmentation result with optical-flow improvement.
        """
        scoremap = output_data[0, 1, :, :]
        scoremap = (scoremap * 255).astype(np.uint8)
        ori_h, ori_w = image.shape[0], image.shape[1]
        evl_h, evl_w = self.eval_size[0], self.eval_size[1]
        # optical flow processing
        cur_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray, (evl_w, evl_h))
        optflow_map = optflow_handle(cur_gray, scoremap, False)
        optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
        optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
        optflow_map = cv2.resize(optflow_map, (ori_w, ori_h))
        optflow_map = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
        bg_im = np.ones_like(optflow_map) * 255
        comb = (optflow_map * image + (1 - optflow_map) * bg_im).astype(np.uint8)
        return comb

    def run_predict(self, image):
        """Run Predicting on an opencv image object.
        Preprocess the image, do inference, and then postprocess the infering output.
        Args:
             image: A valid opencv image object.
        Returns:
             The segmentation result which represents as an opencv image object.
        """
        im_mat = self.preprocess(image)
        im_tensor = fluid.core.PaddleTensor(im_mat.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[0]
        output_data = output_data.as_ndarray()
        return self.postprocess(image, output_data)


def predict_image(seg, image_path):
    """Do Predicting on a image file.
    Decoding the image file and do predicting on it.
    The result will be saved as `result.jpeg`.
    Args:
        seg: The HumanSeg Object which holds a inference model.
            Do preprocessing / predicting / postprocessing on a input image object.
        image_path: Path of the image file needs to be processed.
    """
    img_mat = cv2.imread(image_path)
    img_mat = seg.run_predict(img_mat)
    cv2.imwrite('result.jpeg', img_mat)


def predict_video(seg, video_path):
    """Do Predicting on a video file.
    Decoding the video file and do predicting on each frame.
    All result will be saved as `result.avi`.
    Args:
        seg: The HumanSeg Object which holds a inference model.
            Do preprocessing / predicting / postprocessing on a input image object.
        video_path: Path of a video file needs to be processed.
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
    # Start capturing from video
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
    """Do Predicting on a camera video stream.
    Capturing each video frame from camera and do predicting on it.
    All result frames will be shown in a GUI window.
    Args:
        seg: The HumanSeg Object which holds a inference model.
            Do preprocessing / predicting / postprocessing on a input image object.
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
    """Real Entrypoint of the script.
    Load the human segmentation inference model and do predicting on the input resource.
    Support three types of input: camera stream / video file / image file.
    Args:
      args: The command-line args for inference model.
           Open camera and do predicting on camera stream while `args.use_camera` is true.
           Open the video file and do predicting on it while `args.video_path` is valid.
           Open the image file and do predicting on it while `args.img_path` is valid.
    """
    model_dir = args.model_dir
    use_gpu = args.use_gpu

    # Init model
    mean = [104.008, 116.669, 122.675]
    scale = [1.0, 1.0, 1.0]
    eval_size = (192, 192)
    seg = HumanSeg(model_dir, mean, scale, eval_size, use_gpu)
    if args.use_camera:
        # if enable input video stream from camera
        predict_camera(seg)
    elif args.video_path:
        # if video_path valid, do predicting on the video
        predict_video(seg, args.video_path)
    elif args.img_path:
        # if img_path valid, do predicting on the image
        predict_image(seg, args.img_path)


def parse_args():
    """Parsing command-line argments
    """
    parser = argparse.ArgumentParser('Realtime Human Segmentation')
    parser.add_argument('--model_dir',
                        type=str,
                        default='',
                        help='path of human segmentation model')
    parser.add_argument('--img_path',
                        type=str,
                        default='',
                        help='path of input image')
    parser.add_argument('--video_path',
                        type=str,
                        default='',
                        help='path of input video')
    parser.add_argument('--use_camera',
                        type=bool,
                        default=False,
                        help='input video stream from camera')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=False,
                        help='enable gpu')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
