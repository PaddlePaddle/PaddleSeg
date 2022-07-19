English | [简体中文](video.md)

# Interactive Video Object Segmentation and 3D Medical Imaging Annotation

The following contents are related to interactive video annotation in EISeg, mainly including model selection, data preparation and instructions.

## Environment Configuration

The VTK package should be additionally installed for 3D visualization, please try the following:

```
pip install vtk
```

## Demo

![dance](https://user-images.githubusercontent.com/35907364/175504795-d41f0842-cb18-4675-9763-3e817f168edf.gif)

## Model Selection
Interactive video object segmentation is based on EISeg interactive segmentation algorithms and [MIVOS](https://github.com/hkchengrex/MiVOS) algorithm. It is an efficient image and video annotation software based on PaddlePaddle.
EISeg 1.0 covers high-quality interactive video object segmentation models in different directions such as general, liver, CT spinal structures, etc., which is convenient for developers to quickly annotate videos and reduce the cost of annotation. For 3D Medical Imaging Annotation, we regard medical slice data as the video frames, and realize the labeling of 3D medical images by using video annotation method.
Before using EISeg, please download the propagation model parameters. If you want to use the 3D display function, you can check the 3D display function in the `Display` menu.

![lits](https://user-images.githubusercontent.com/35907364/178422205-40327d43-c7d4-4a5d-87fb-63c08308fb9f.gif)


| Model Type  | Applicable Scenarios                                    | Model Architecture       | Download Link                                                     | Corresponding Propagation Model Download Link                  |
| -------- |---------------------------------------------------------| -------------- | ------------------------------------------------------------ |----------------------------------------------------------------|
| High Performance Model | Image annotation in generic scenarios                   | HRNet18_OCR64  | [static_hrnet18_ocr64_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_cocolvis.zip) | [static_propagation]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)       |
| Lightweight Model | Image annotation in generic scenarios                   | HRNet18s_OCR48 | [static_hrnet18s_ocr48_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_cocolvis.zip) | [static_propagation]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)       |
| High Performance Model | Image annotation in generic scenarios                   | EdgeFlow | [static_edgeflow_cocolvis](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_edgeflow_cocolvis.zip) | [static_propagation]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)       |
| High Performance Model | Annotation in portrait scenarios                        | HRNet18_OCR64  | [static_hrnet18_ocr64_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18_ocr64_human.zip) | [static_propagation]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)       |
| Lightweight Model | Annotation in portrait scenarios                        | HRNet18s_OCR48 | [static_hrnet18s_ocr48_human](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_human.zip) | [static_propagation]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)       |
| Lightweight Model | Annotation of liver in medical scenarios                | HRNet18s_OCR48 | [static_hrnet18s_ocr48_lits](https://paddleseg.bj.bcebos.com/eiseg/0.4/static_hrnet18s_ocr48_lits.zip) | [static_propagation_lits]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856)  |
| Lightweight Model | Annotation of CT Spinal Structures in medical scenarios | HRNet18s_OCR48 | [static_hrnet18s_ocr48_MRSpineSeg](https://paddleseg.bj.bcebos.com/eiseg/0.5/static_hrnet18s_ocr48_MRSpineSeg.zip) | [static_propagation_spine]( https://app.mopinion.com/survey/public/take-survey/7b29c771b228bbf2512d1c5f9ec784e4b861f856) |

## Data Preparation

- Due to the large computation in video segmentation, it is recommended to use the compyter with  a graphics card,  and the number of labeled video should not exceed 100 frames. If the video exceeds the number of frames, you can use [cut_video.py](../tool/cut_video.py) to cut video.
- 3D medical imaging annotation is based on interactive video segmentation, so please convert the sliced images into mp4 format before labeling. The script is: [medical2video.py](../tool/medical2video.py).

## Using

After opening the software, make the following settings before annotating:

1. **Load Model Parameter**

   Select the appropriate network and load the corresponding model parameters. After downloading and decompressing the right model and parameters, the model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory, and only the location of the model parameter at the end of `*.pdiparams`need to be selected when loading the model. The initialization of the static model takes some time, please wait patiently until the model is loaded. The correctly loaded model parameters will be recorded in `Recent Model Parameters`, which can be easily switched, and the exiting model parameter will be loaded automatically the next time you open the software.

2. **Load Image**

   Open the image or image folder. Things go well when you see that the main screen image is loaded correctly and the image path is rightly shown in `Data List`.

3. **Add/Load Label**

   Add/load labels. New labels can be created by `Add Label`, which are divided into 4 columns corresponding to pixel value, description, color and deletion. The newly created labels can be saved as txt files by `Save Label List`, and other collaborators can import labels by `Load Label List`. Labels imported by loading will be loaded automatically after restarting the software.

4. **Load Propagation Model Parameter**

    Select the corresponding propagation model parameters. After downloading and decompressing the right model and parameters, the model structure `*.pdmodel` and the corresponding model parameters `*.pdiparams` should be put into the same directory, and only one of the location of the model parameter at the end of `*.pdiparams`need to be selected when loading the model.  The other models will be loaded automatically.

5. **Annotate Referance frame**

    During interactive annotation, users add positive and negative points with left and right mouse clicks, respectively. After finishing interactive segmentation, you can push Space button and the tool generates a polygon frame around the target border. Users can
adjust the polygon vertexes to further improve segmentation accuracy.

6. **Autosave**

   You can choose the right folder and have the `autosave` set up, so that the annotated image will be saved automatically when switching images.
