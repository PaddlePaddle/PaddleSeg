import os

import cv2


def video_cut(video_path, save_path, cut_length):
    video_capture = cv2.VideoCapture(video_path)
    # FPS
    fps = video_capture.get(5)
    print(video_capture.isOpened())
    print("fps", video_capture.get(5))
    print("COUNT", video_capture.get(7))
    size = (int(video_capture.get(3)), int(video_capture.get(4)))
    frame_index = 0
    flag = 0
    success, bgr_image = video_capture.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    final_path = os.path.join(save_path,
                              str(frame_index // cut_length) + '.mp4')
    v = cv2.VideoWriter(final_path, fourcc, fps, size)
    while success:
        if v.isOpened():
            v.write(bgr_image)

        if frame_index == cut_length * flag + cut_length:
            if v.isOpened():
                v.release()

        if frame_index == cut_length * flag:
            final_path = os.path.join(save_path,
                                      str(frame_index // cut_length) + '.mp4')
            v = cv2.VideoWriter(final_path, fourcc, fps, size)
            flag += 1
        success, bgr_image = video_capture.read()
        frame_index = frame_index + 1

    video_capture.release()
    v.release()


if __name__ == "__main__":
    video_path = '/PATH/TO/VIDEO'
    save_path = "/PATH/TO/SAVE"
    cut_length = 50
    video_cut(video_path, save_path, cut_length)
