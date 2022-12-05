
# 2D Image Detection Anotation

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/35907364/203893714-9a1102d7-3977-4f03-8be2-5d6f7728211c.gif" align="middle" alt="LOGO" width = "800" />
</p>
</div>

The following content is about how to use EISeg to annotate 2D images. Model preparation and how to use can be seen as follow:

## Model Preparation

Before using EISeg for det annotation, if you use the pre annotation function, please download the detection pretrained model firstly. EISeg 1.1.0 provides PicoDet-S model trained on COCO dataset to meet the requirements of generic detection scenarios. Users can choose manual annotation or rely on AI model to generate labels.


| Model Type   | Applicable Scenarios                   | Model Architecture     | Download Link                                                     |
| ---------- | -------------------------- |---------------------| ------------------------------------------------------------ |
| Lightweight Model | Image annotation in generic scenarios           | PicoDet-S           | [PicoDet-S ](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar) |


**NOTE**： The downloaded model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory. In the det annotation mode, if the pre annotation function is enabled, you only need to decide the location of the model parameter at the end of `*.pdiparams`, and `*.pdmodel` will be loaded automatically. If you choose not to enable pre annotation, you do not need to load the model.

## Using

The overall process of using EISeg detection labeling is shown in the figure below. For details, please refer to the following instructions:

<p align="center">
<img src="https://user-images.githubusercontent.com/48357642/202401769-49f3de96-41b8-406b-ae1c-b6edf163a5c4.png" width="100%" height="100%">
<p align="center">
 The overall process
</p>



After opening the software, make the following settings before annotating:

0. **Annotation Mode Selection**

   Select the detection annotation mode. Click the mode switch button in the upper right corner of EISeg to enter the detection annotation mode, and select whether to enable the detection pre annotation function according to the pop-up prompt. If not, skip steps 1 and 2 below.

1. **Load Model Parameter**

   Select the appropriate network and load the corresponding model parameters. Click the `Load Network Parameters` button in the `Model Selection` area，After downloading and decompressing the right model and parameters, the model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory, and only the location of the model parameter at the end of `*.pdiparams`need to be selected when loading the model. The initialization of the static model takes some time, please wait patiently until the model is loaded. The correctly loaded model parameters will be recorded in `Recent Model Parameters`, which can be easily switched, and the exiting model parameter will be loaded automatically the next time you open the software.
   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202402532-ef022aaa-eb43-4b1d-a2f8-7f4b0380bd5f.png" width="40%" height="20%">
   </p>

2. **Pre Annotation Settings**

   Create labels correspondence. Click the `Pre Annotation Settings` button in the `Detection Settings` area to set the corresponding relationship between pre annotation model labels and user-defined labels in the pop-up interface, as well as whether to enable labels of a certain category. In addition, you can also search for specific labels in the search box of the interface, so that users can quickly select target labels to establish the corresponding relationship between labels. By default, COCO 80 labels will be enabled as pre annotation model labels. After establishing the label correspondence, only the user enabled label pairs and user-defined label names will be displayed in the image pre annotation results.
   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202402889-f45d3275-645e-4633-898a-d60b8c025f19.png" width="40%" height="20%">
   </p>

   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202403435-5c30bf95-1389-4e63-8a49-cb08524050e6.png" width="40%" height="20%">
   </p>

3. **Load Image**

   Open the image or image folder. Things go well when you see that the main screen image is loaded correctly and the image path is rightly shown in `Data List`.

   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202403076-4d67bbc8-b9d8-401f-811d-cf5c7f8f8a1c.png" width="40%" height="20%">
   </p>

4. **Add/Load Label**

   Add/load labels. If the user has enabled the pre annotation function, the labels predicted by the model will be automatically added to the label list. Users can also manually click the `Add Label` button to create a new label, which are divided into 4 columns corresponding to pixel value, description, color and deletion. The newly created labels can be saved as txt files by `Save Label List`, and other collaborators can import labels by `Load Label List`. Labels imported by loading will be loaded automatically after restarting the software. In addition, label search is also supported. You can search whether there is a specific label in the label list through `Search`, so that users can quickly locate the target label.
   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202403524-6a008e5f-9f00-418c-b91e-0602ae71fb49.png" width="40%" height="20%">
   </p>

5. **Annotation**

   Start labeling. If you enable the pre annotation function, you can modify and delete the bounding box predicted by the model. You can also click the `Start Draw Function` button in the `Detection Settings` area, and select a specific label in the `Label List` area to manually draw bounding box in the currently displayed image. You can drag the upper left and lower right corners of the bounding box to modify the box. You can click the left mouse button to select the specified bounding box to delete, or you can select the color of the cross hairs when drawing the bounding box. In addition, the color of the bounding box corresponds to the color of the label.
   <p align="center">
   <img src="https://user-images.githubusercontent.com/48357642/202403616-8a57bf25-7eee-4814-be57-0d7889042ebb.png" width="40%" height="20%">
   </p>

6. **Autosave**

   You can choose the right folder and have the `autosave` set up, so that the annotated image will be saved automatically when switching images.

## Instruction of New Functions

- **Detection Settings**

    - After enabling the pre annotation function, the user can drag the `score threshold` slider in the `Detection Settings` area to set the confidence level of the model prediction, thus displaying different numbers of bounding boxes. The manually edited boxes will not disappear with the sliding of the slider.
    - After enabling the pre annotation function, the user can click the `ReDet` button in the `Detection Settings` area to delete all the original bounding boxes and re inference and display them.

- **Save Format**

    - By default, Annotation results are saved in COCO format. You can also select VOC and YOLO formats according to your needs.
    - With no specified save path, the image is save to the label folder under the current image folder by default.
