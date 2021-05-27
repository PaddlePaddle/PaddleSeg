# 数据格式转换

目前PaddleSeg暂不支持医学格式三维图像(如Dicom、nii、nii.gz等)作为直接输入，可按如下代码转换成jpg、png格式。

```    for i in range(seg_array.shape[0]):
import SimpleITK 
import cv2
origin_path = origin.nii.gz
seg_path = seg.nii.gz
origin_array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(origin_path))
seg_array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(seg_path))
for i in range(origin_array.shape[0]):
    cv2.imwrite('image' + str(i) + '.jpg', origin_array[i,:,:])
    cv2.imwrite('label' + str(i) + '.png', seg_array[i,:,:])
```



