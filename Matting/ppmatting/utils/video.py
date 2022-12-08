import os
import warnings

import cv2
import paddle
import numpy as np

import ppmatting
import ppmatting.transforms as T


class VideoReader(paddle.io.Dataset):
    """
    Read a video
    """

    def __init__(self, path, transforms=None):
        super().__init__()
        if not os.path.exists(path):
            raise IOError('There is not found about video path:{} '.format(
                path))
        self.cap_video = cv2.VideoCapture(path)
        if not self.cap_video.isOpened():
            raise IOError('Video can not be oepned normally')

        # Get some video property
        self.fps = int(self.cap_video.get(cv2.CAP_PROP_FPS))
        self.frames = int(self.cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        transforms = [] if transforms is None else transforms
        if transforms is None or isinstance(transforms, list):
            self.transforms = T.Compose(transforms)
        elif isinstance(transforms, T.Compose):
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms type is error, it should be list or ppmatting,transforms.Compose"
            )

    def __len__(self):
        return self.frames

    def __getitem__(self, idx):
        if idx >= self.frames:
            raise IndexError('The frame {} is read failed.'.format(idx))
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap_video.retrieve()
        if not ret:
            warnings.warn(
                "the frame {} is read failed. Video reading exit.".format(idx))
            raise IndexError('The frame {} is read failed.'.format(idx))

        data = {'img': frame}
        if self.transforms is not None:
            data = self.transforms(data)
        data['ori_img'] = frame.transpose((2, 0, 1)) / 255.

        return data

    def release(self):
        self.cap_video.release()


class VideoWriter:
    """
    Video writer.

    Args:
        path (str): The path to save a video.
        fps (int): The fps of the saved video.
        frame_size (tuple): The frame size (width, height) of the saved video.
        is_color (bool): Whethe to save the video in color format.
    """

    def __init__(self, path, fps, frame_size, is_color=True):
        self.is_color = is_color

        ppmatting.utils.mkdir(path)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap_out = cv2.VideoWriter(
            filename=path,
            fourcc=fourcc,
            fps=fps,
            frameSize=frame_size,
            isColor=is_color)

    def write(self, frames):
        """ 
        Save frames.

        Args:
            frames(Tensor|numpy.ndarray): If `frames` is a tensor, it's shape should be like [N, C, H, W].
                If it is a ndarray, it's shape should be like [H, W, 3] or [H, W]. The value is in [0, 1].
        """
        if isinstance(frames, paddle.Tensor):
            if frames.ndim != 4:
                raise ValueError(
                    'The frames should have the shape like [N, C, H, W], but it is {}'.
                    format(frames.shape))
            n, c, h, w = frames.shape
            if not (c == 1 or c == 3):
                raise ValueError(
                    'the channels of frames should be 1 or 3, but it is {}'.
                    format(c))
            if c == 1 and self.is_color:
                frames = paddle.repeat_interleave(frames, repeats=3, axis=1)

            frames = (frames.transpose(
                (0, 2, 3, 1)).numpy() * 255).astype('uint8')
            for i in range(n):
                frame = frames[i]
                self.cap_out.write(frame)
        else:
            frames = (frames * 255).astype('uint8')
            self.cap_out.write(frames)

    def release(self):
        self.cap_out.release()
