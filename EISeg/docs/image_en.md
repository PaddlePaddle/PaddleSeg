English | [简体中文](image.md)
# 2D Image Anotation

The following content is about how to use EISeg to annotate 2D images. Model preparation and how to use can be seen as follow:

## Model Preparation

Please download the model parameters before using EIseg. EISeg 0.5.0 provides seven direction models trained on COCO+LVIS, large-scale portrait data, mapping_challenge, MRSpineSeg, Chest Xray, LiTS and Self-built aluminum plate quality inspection data set to meet the labeling needs of generic and portrait scenarios as well as architecture, medical and industrial images. The model architecture corresponds to the network selection module in EISeg interactive tools, and users need to select different network structures and loading parameters in accordance with their own needs.

| Model Type          | Applicable Scenarios                     | Model Architecture | Download Link                                                |
|---------------------| ---------------------------------------- | ------------------ | ------------------------------------------------------------ |
| High Performance Model | Image annotation in generic scenarios    | HRNet18_OCR64      | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip) |
| Lightweight Model   | Image annotation in generic scenarios    | HRNet18s_OCR48     | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip) |
| High Performance Model | Annotation in portrait scenarios         | HRNet18_OCR64      | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip) |
| High Performance Model | Image annotation in generic scenarios    | EdgeFlow           | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip) |
| Lightweight Model   | Annotation in portrait scenarios         | HRNet18s_OCR48     | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip) |
| Lightweight Model   | Annotation of remote sensing building    | HRNet18s_OCR48     | [static_hrnet18_ocr48_rsbuilding_instance](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr48_rsbuilding_instance.zip) |
| High Performance Model | Annotation of chest Xray in medical scenarios | Resnet50_DeeplabV3+   | [static_resnet50_deeplab_chest_xray \*](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_resnet50_deeplab_chest_xray.zip) |
| Lightweight Model   | Annotation of liver in medical scenarios | HRNet18s_OCR48     | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) |
| Lightweight Model   | Annotation of Spinal Structures in medical scenarios | HRNet18s_OCR48     | [static_hrnet18s_ocr48_MRSpineSeg](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_MRSpineSeg.zip) |
| Lightweight Model   | Annotation of Aluminum plate defects in industrial scenarios | HRNet18s_OCR48     | [static_hrnet18s_ocr48_aluminium ](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_aluminium.zip) |

**NOTE**： The downloaded model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory. When loading the model, you only need to decide the location of the model parameter at the end of `*.pdiparams`, and `*.pdmodel` will be loaded automatically. When using `EdgeFlow` model, please turn off `Use Mask`, and check `Use Mask` when adopting other models. For `High Performance Model`, we recommend to utilize the computer with gpu for a smoother annotation experience.

## Using

After opening the software, make the following settings before annotating:

1. **Load Model Parameter**

   Select the appropriate network and load the corresponding model parameters.  After downloading and decompressing the right model and parameters, the model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory, and only the location of the model parameter at the end of `*.pdiparams`need to be selected when loading the model. The initialization of the static model takes some time, please wait patiently until the model is loaded. The correctly loaded model parameters will be recorded in `Recent Model Parameters`, which can be easily switched, and the exiting model parameter will be loaded automatically the next time you open the software.

2. **Load Image**

   Open the image or image folder. Things go well when you see that the main screen image is loaded correctly and the image path is rightly shown in `Data List`.

3. **Add/Load Label**

   Add/load labels. New labels can be created by `Add Label`, which are divided into 4 columns corresponding to pixel value, description, color and deletion. The newly created labels can be saved as txt files by `Save Label List`, and other collaborators can import labels by `Load Label List`. Labels imported by loading will be loaded automatically after restarting the software.

4. **Annotation**

    During interactive annotation, users add positive and negative points with left and right mouse clicks, respectively. After finishing interactive segmentation, you can push Space button and the tool generates a polygon frame around the target border. Users can
adjust the polygon vertexes to further improve segmentation accuracy.

5. **Autosave**

   You can choose the right folder and have the `autosave` set up, so that the annotated image will be saved automatically when switching images.

Start the annotation when the above are all set up. Here are the commonly used keys/shortcut keys by default, press `E` to modify them as you need.

| Keys/Shortcut Keys                | Function                       |
| --------------------------------- | ------------------------------ |
| Left Mouse Button                 | Add Positive Sample Points     |
| Right Mouse Button                | Add Negative Sample Points     |
| Middle Mouse Button               | Image Panning                  |
| Ctrl+Middle Mouse Button（wheel） | Image Zooming                  |
| S                                 | Previous Image                 |
| F                                 | Next Image                     |
| Space                             | Finish Annotation/Switch State |
| Ctrl+Z                            | Undo                           |
| Ctrl+Shift+Z                      | Clear                          |
| Ctrl+Y                            | Redo                           |
| Ctrl+A                            | Open Image                     |
| Shift+A                           | Open Folder                    |
| E                                 | Open Shortcut Key List         |
| Backspace                         | Delete Polygon                 |
| Double Click（point）             | Delete Point                   |
| Double Click（edge）              | Add Point                      |

## Instruction of New Functions

- **Polygon**

  - Click Space key to complete interactive annotation, then appears the polygon boundary.
  - When you need to continue the interactive process inside the polygon, click Space to switch to interactive mode so the polygon cannot be selected and changed.
  - The polygon can be deleted. Use the left mouse to drag the anchor point, double-click the anchor point to delete it, and double-click a side to add an anchor point.
  - With `Keep Maximum Connected Blocks` on, only the largest area will remain in the image, the rest of the small areas will not be displayed and saved.

- **Save Format**

  - Polygons will be recorded and automatically loaded after setting `JSON Save` or `COCO Save`.
  - With no specified save path, the image is save to the label folder under the current image folder by default.
  - If there are images with the same name but different suffixes, you can open `labels and images with the same extensions`.
  - You can also save as grayscale,  pseudo-color or matting image, see tools 7-9 in the toolbar.

- **Generate mask**

  - Labels can be dragged by holding down the second column, and the final generated mask will be overwritten from top to bottom according to the label list.

- **Interface Module**

  - You can select the interface module to be presented in `Display`, and the normal exit status and location of the interface module will be recorded, and loaded automatically when you open it next time.

- **Vertical Segmentation**

  EISeg now supports remote sensing images and medical images segmentation, and additional dependencies need to be installed for their functioning.

  - Install GDAL for remote sensing image segmentation, please refer to [Remote Sensing Segmentation](docs/remote_sensing_en.md)。
  - Install SimpleITK for medical images segmentation, please refer to [Medical Image Segmentation](docs/medical_en.md)。

- **Scripting Tool**

  EISeg currently provides scripting tools including annotation to PaddleX dataset, delineation of COCO format and semantic labels to instance labels, etc. See [Scripting Tools Usage](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/EISeg/) for more details.
