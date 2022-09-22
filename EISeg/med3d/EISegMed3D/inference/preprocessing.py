import numpy as np
import SimpleITK as sitk


def resampleImage(refer_image,
                  out_size,
                  out_spacing=None,
                  interpolator=sitk.sitkLinear):
    # 根据输出图像，对SimpleITK 的数据进行重新采样。重新设置spacing和shape
    if out_spacing is None:
        out_spacing = tuple((refer_image.GetSize() / np.array(out_size)) *
                            refer_image.GetSpacing())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(refer_image)
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(refer_image), out_spacing


def crop_wwwc(sitkimg, max_v, min_v):
    # 对SimpleITK的数据进行窗宽窗位的裁剪，应与训练前对数据预处理时一致
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(max_v)
    intensityWindow.SetWindowMinimum(min_v)
    return intensityWindow.Execute(sitkimg)


def GetLargestConnectedCompont(binarysitk_image):
    # 最大连通域提取,binarysitk_image 是掩膜
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)  # 根据掩膜计算统计量
    # stats.
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():  # 掩膜中存在的标签类别
        size = stats.GetPhysicalSize(l)
        if maxsize < size:  # 只保留最大的标签类别
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    if len(stats.GetLabels()):
        outmask[labelmaskimage == maxlabel] = 255
        outmask[labelmaskimage != maxlabel] = 0
    return outmask
