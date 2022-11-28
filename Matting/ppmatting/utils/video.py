import os

import cv2
import paddle

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
        self.transforms = T.Compose(transforms)

    def __len__(self):
        return self.frames

    def __getitem__(self, idx):
        if idx >= self.frames:
            raise IndexError('The frame {} is read failed.'.format(idx))
        self.cap_video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap_video.retrieve()
        if not ret:
            raise IOError('The frame {} is read failed.'.format(idx))

        data = {'img': frame}
        if self.transforms is not None:
            data = self.transforms(data)

        return data


if __name__ == "__main__":
    import time
    transforms = [T.Normalize(mean=[0, 0, 0], std=[1, 1, 1])]
    vr = VideoReader(
        path='/ssd1/home/chenguowei01/github/PaddleSeg/Matting/test/test_real_video/img_9652.mov',
        transforms=transforms)
    print(len(vr))
    for i, data in enumerate(vr):
        print(data.keys())
        print(data['img'].max(), data['img'].min())
        # print(f.sum(), end=' ')
    print(vr.fps, vr.frames, vr.width, vr.height)
