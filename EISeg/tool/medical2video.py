import os.path as osp

import SimpleITK as sitk
import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
from tqdm import tqdm


def load_medical_data(f):
    """
    load data of different format into numpy array, return data is in xyz
    f: the complete path to the file that you want to load
    """
    filename = osp.basename(f).lower()

    if filename.endswith((".nii", ".nii.gz", ".dcm")):
        itkimage = sitk.ReadImage(f)
        if itkimage.GetDimension() == 4:
            slicer = sitk.ExtractImageFilter()
            s = list(itkimage.GetSize())
            s[-1] = 0
            slicer.SetSize(s)
            images = []
            for slice_idx in range(itkimage.GetSize()[-1]):
                slicer.SetIndex([0, 0, 0, slice_idx])
                sitk_volume = slicer.Execute(itkimage)
                images.append(sitk_volume)
            images = [sitk.DICOMOrient(img, "SLP") for img in images]
            f_nps = [sitk.GetArrayFromImage(img) for img in images]
        else:
            image = sitk.DICOMOrient(itkimage, "SLP")
            f_np = sitk.GetArrayFromImage(image)
            f_nps = [f_np]

    elif filename.endswith((".mha", ".mhd", "nrrd")):
        itkimage = sitk.DICOMOrient(sitk.ReadImage(f), "SLP")
        f_np = sitk.GetArrayFromImage(itkimage)
        if f_np.ndim == 4:
            f_nps = [f_np[:, :, :, idx] for idx in range(f_np.shape[3])]
        else:
            f_nps = [f_np]
    elif filename.endswith(".raw"):
        raise RuntimeError(
            f"Received {f}. Please only provide path to .mhd file, not to .raw file"
        )
    else:
        raise NotImplementedError

    return f_nps


def normalize(frame, ww=400, wc=0):
    wl = wc - ww / 2
    wh = wc + ww / 2
    frame = frame.astype("float16")
    np.clip(frame, wl, wh, out=frame)
    frame = (frame - wl) / ww * 255
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def array_to_video(array_data, video_path, fps=15):
    h, w, s = array_data.shape
    fourcc = VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for idx in tqdm(range(s)):
        frame = array_data[:, :, idx]
        frame = normalize(frame)
        videoWriter.write(frame)

    videoWriter.release()
    cv2.destroyAllWindows()
    print("转视频结束！")


if __name__ == "__main__":
    path = "/home/lin/Desktop/data/coronacases_org_001.nii.gz"
    video_path = "/home/lin/Desktop/temp.mp4"
    total_data = load_medical_data(path)
    for data in total_data:
        array_to_video(data, video_path)
