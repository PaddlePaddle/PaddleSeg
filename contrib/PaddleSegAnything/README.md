# Segment Anything with PaddleSeg

## Contents
1. Overview
2. News
2. Performance
3. Try it by yourself with one line of code

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Alexander Kirillov](https://alexander-kirillov.github.io/), [Eric Mintun](https://ericmintun.github.io/), [Nikhila Ravi](https://nikhilaravi.com/), [Hanzi Mao](https://hanzimao.me/), Chloe Rolland, Laura Gustafson, [Tete Xiao](https://tetexiao.com), [Spencer Whitehead](https://www.spencerwhitehead.com/), Alex Berg, Wan-Yen Lo, [Piotr Dollar](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

[[`Paper`](https://ai.facebook.com/research/publications/segment-anything/)] [[`Project`](https://segment-anything.com/)] [[`Demo`](https://segment-anything.com/demo)] [[`Dataset`](https://segment-anything.com/dataset/index.html)] [[`Blog`](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/)] [[`BibTeX`](#citing-segment-anything)]

**Segment Anything Model (SAM)**: a new task, model, and dataset for image segmentation. We implemente it with the PaddlePaddle framework. SAM can produce high quality object masks from different types of prompts including points, boxes, masks and text. Further, SAM can generate masks for all objects in whole image. It built a largest segmentation [dataset](https://segment-anything.com/dataset/index.html) to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. SAM has impressive zero-shot performance on a variety of tasks, even often competitive with or even superior to prior fully supervised results.

## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

<div align="center">
<img src="https://github.com/Sunting78/images/blob/master/sam_new.gif"  width="1000" />
</div>

## <img src="https://user-images.githubusercontent.com/34859558/188439970-18e51958-61bf-4b43-a73c-a9de3eb4fc79.png" width="25"/> Try it by yourself with one line of code

### Preparation
* Install PaddlePaddle and relative environments based on the [installation guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).
* Install PaddleSeg based on the [reference](../../docs/install.md).
* Clone the PaddleSeg reporitory:
    ```bash
    git clone https://github.com/PaddlePaddle/PaddleSeg.git
    cd PaddleSeg
    pip install -r requirements.txt
    ```
* Download the example image to ```contrib/PaddleSegAnything/examples```, and the file structure is as following:
    ```bash
    wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
    ```

    ```
    PaddleSeg/contrib
    ├── PaddleSegAnything
    │   ├── examples
    │   │   └──  cityscapes_demo.png
    │   ├── modeling
    │   └── script

    ```

### Start the gradio service.
In this step, we start a gradio service with the following scrip on local machine and you can try out our project with your own images.

1. Run the following script:

```bash
python script/amg_paddle.py --model-type [vit_l/vit_b/vit_h] # default is vit_h

```
Note:
*  There are three model options for you, vit_b, vit_l and vit_h, represent vit_base, vit_large and vit_huge. Large model is more accurate and also slower. You can choose the model size based on your device.
* The test result shows that vit_h needs 16G video memory and needs around 10s to infer an image on V100.

2. Open the webpage on your localhost: ```http://0.0.0.0:8017```

3. Try it out by clear and upload the test image! Our example looks like:

    <div align="center">
    <img src="https://user-images.githubusercontent.com/34859558/230873989-9597527e-bef6-47ce-988b-977198794d75.jpg"  width = "1000" />  
    </div>
