# Segment Anything with PaddleSeg

## Reference

> Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick. [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/).


## Contents
1. Overview
2. Performance
3. Try it by yourself with one line of code

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview

We implemente the segment anything with the PaddlePaddle framework. **Segment Anything Model (SAM)** is a new task, model, and dataset for image segmentation.  It can produce high quality object masks from different types of prompts including points, boxes, masks and text. Further, SAM can generate masks for all objects in whole image. It built a largest segmentation [dataset](https://segment-anything.com/dataset/index.html) to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. SAM has impressive zero-shot performance on a variety of tasks, even often competitive with or even superior to prior fully supervised results.

We provide the pretrained model parameters of PaddlePaddle format, including [vit_b](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams), [vit_l](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams) and [vit_h](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams).

## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

<div align="center">
<img src="https://github.com/Sunting78/images/blob/master/sam_new.gif"  width="1000" />
</div>


## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> Try it by yourself with one line of code

### 1. Preparation
* Install PaddlePaddle and relative environments based on the [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).
* Install PaddleSeg based on the [reference](../../docs/install.md).
* Clone the PaddleSeg reporitory:
    ```bash
    git clone https://github.com/PaddlePaddle/PaddleSeg.git
    cd PaddleSeg
    pip install -r requirements.txt
    ```
* Download the example image to ```contrib/SegmentAnything/examples```, and the file structure is as following:
    ```bash
    wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
    ```

    ```
    PaddleSeg/contrib
    ├── SegmentAnything
    │   ├── examples
    │   │   └──  cityscapes_demo.png
    │   ├── segment_anything
    │   └── scripts

    ```

### 2. Segment the whole image on webpage.
In this step, we start a gradio service with the following scrip on local machine and you can try out our project with your own images.

1. Run the following script:
    ```bash
    python scripts/amg_paddle.py --model-type [vit_l/vit_b/vit_h] # default is vit_h

    ```
    Note:
    *  There are three model options for you, vit_b, vit_l and vit_h, represent vit_base, vit_large and vit_huge. Large model is more accurate and also slower. You can choose the model size based on your device.
    * The test result shows that vit_h needs 16G video memory and needs around 10s to infer an image on V100.

2. Open the webpage on your localhost: ```http://0.0.0.0:8017```

3. Try it out by clear and upload the test image! Our example looks like:

    <div align="center">
    <img src="https://user-images.githubusercontent.com/34859558/230873989-9597527e-bef6-47ce-988b-977198794d75.jpg"  width = "1000" />  
    </div>

### 3. Segment the object with prompts
You can run the following commands to produce masks from different types of prompts including points, boxes, and masks, as follow:


1. Box prompt

```bash
python scripts/promt_predict.py --input_path xxx.png --box_prompt 1050 370 1500 700 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

2. Point prompt
```bash
python scripts/promt_predict.py --input_path xxx.png --point_prompt 1200 450 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

3. Mask prompt
```bash
python scripts/promt_predict.py --input_path xxx.png --mask_prompt xxx.png --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

Note:
* mask_prompt is the path of a binary image.
