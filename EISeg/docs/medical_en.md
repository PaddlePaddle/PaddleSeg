# Medical Treatment

This part presents documents related to medical treatment in EISeg, including its environment configuration and functions.

## 1 Environment Configuration

The SimpleITK package should be additionally installed for image reading, please try the following:

```
pip install SimpleITK
```

## 2 Functions

EISeg can open **single-layer Dicom format images**, while the support for Nitfi format and multiple Dicom is under development. EISeg fines the image format by its expansion name. To open a single image you need to select Medical Image in the drop-down menu of type at the bottom right corner, as shown below.

The folder and natural image share the same process. When opening an image with a .dcm suffix, you will be asked whether to turn on the medical component.

[![med-prompt](https://camo.githubusercontent.com/ba9ab11d3e602ae61769d2926bd6774d1dfa633346cc483ab04bf4c89e65d2d0/68747470733a2f2f6c696e68616e6465762e6769746875622e696f2f6173736574732f696d672f706f73742f4d65642f6d65642d70726f6d70742e706e67)](https://camo.githubusercontent.com/ba9ab11d3e602ae61769d2926bd6774d1dfa633346cc483ab04bf4c89e65d2d0/68747470733a2f2f6c696e68616e6465762e6769746875622e696f2f6173736574732f696d672f706f73742f4d65642f6d65642d70726f6d70742e706e67)



Click Yes and there appears the setting panel of the image window width and position.

[![med-widget](https://camo.githubusercontent.com/05e9c84842f9b18ad94d5a9d7610642607f569d3ef6a9d97fd445a60df9ece46/68747470733a2f2f6c696e68616e6465762e6769746875622e696f2f6173736574732f696d672f706f73742f4d65642f6d65642d7769646765742e706e67)](https://camo.githubusercontent.com/05e9c84842f9b18ad94d5a9d7610642607f569d3ef6a9d97fd445a60df9ece46/68747470733a2f2f6c696e68616e6465762e6769746875622e696f2f6173736574732f696d672f706f73742f4d65642f6d65642d7769646765742e706e67)

The window width and position serve to limit the intensity range for easy observation of the CT scanning. The value stored at each pixel point in the CT scan represents the density of the human body at that location, so the higher the density the larger the value. The data range of the image is usually -1024 to 1024. However, the human eye cannot distinguish 2048 shades of gray when viewing the scan, so a smaller intensity range is usually adopted to increase the grayscale differences of the images within, thus facilitating the observation. This is done by selecting the section ranging from Window - Window Width/2 to Window + Window Width/2, and presenting the data in a 256-grayscale image.

For inference, EISeg provides the [pre-trained model for liver segmentation](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) for medical scenarios, with recommended window widths of 400, 0. This model performs best for liver segmentation and can also be used for other tissues or organs. 
