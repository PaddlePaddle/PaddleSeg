import argparse
import os
import os.path as osp
import cv2
import numpy as np

from utils.humanseg_postprocess import postprocess
import models
import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg inference for video')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for inference',
        type=str)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help=
        'Video path for inference, camera will be used if the path not existing',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')

    return parser.parse_args()


def predict(img, model, test_transforms):
    model.arrange_transform(transforms=test_transforms, mode='test')
    img, im_info = test_transforms(img)
    img = np.expand_dims(img, axis=0)
    result = model.exe.run(
        model.test_prog,
        feed={'image': img},
        fetch_list=list(model.test_outputs.values()))
    score_map = result[1]
    score_map = np.squeeze(score_map, axis=0)
    score_map = np.transpose(score_map, (1, 2, 0))
    return score_map, im_info


def recover(img, im_info):
    keys = list(im_info.keys())
    for k in keys[::-1]:
        if k == 'shape_before_resize':
            h, w = im_info[k][0], im_info[k][1]
            img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        elif k == 'shape_before_padding':
            h, w = im_info[k][0], im_info[k][1]
            img = img[0:h, 0:w]
    return img


def video_infer(args):
    test_transforms = transforms.Compose(
        [transforms.Resize((192, 192)),
         transforms.Normalize()])
    model = models.load_model(args.model_dir)
    if not args.video_path:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file, "
                      "--video_path whether existing: {}"
                      " or camera whether working".format(args.video_path))
        return
    if args.video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 用于保存预测结果视频
        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)
        out = cv2.VideoWriter(
            osp.join(args.save_dir, 'result.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        # 开始获取视频帧
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                score_map, im_info = predict(frame, model, test_transforms)
                img = cv2.resize(frame, (192, 192))
                img_mat = postprocess(img, score_map)
                img_mat = recover(img_mat, im_info)
                bg_im = np.ones_like(img_mat) * 255
                comb = (img_mat * frame + (1 - img_mat) * bg_im).astype(
                    np.uint8)
                out.write(comb)
            else:
                break
        cap.release()
        out.release()

    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                score_map, im_info = predict(frame, model, test_transforms)
                img = cv2.resize(frame, (192, 192))
                img_mat = postprocess(img, score_map)
                img_mat = recover(img_mat, im_info)
                bg_im = np.ones_like(img_mat) * 255
                comb = (img_mat * frame + (1 - img_mat) * bg_im).astype(
                    np.uint8)
                cv2.imshow('HumanSegmentation', comb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    video_infer(args)
