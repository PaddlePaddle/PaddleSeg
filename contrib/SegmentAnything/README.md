# Segment Anything with PaddleSeg


## Contents
1. Overview
2. Performance
3. Try it by yourself with one line of code
4. Reference


## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview

We implemente the segment anything with the PaddlePaddle framework. **Segment Anything Model (SAM)** is a new task, model, and dataset for image segmentation. It built a largest segmentation [dataset](https://segment-anything.com/dataset/index.html) to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. Further, SAM can produce high quality object masks from different types of prompts including points, boxes, masks and text. SAM has impressive zero-shot performance on a variety of tasks, even often competitive with or even superior to prior fully supervised results. However, the SAM model based on text prompt is not released at the moment. Therefore, we use a combination of **SAM** and **CLIP** to calculate the similarity between the output masks and text prompt. In this way, you can use **text prompt** to segment anything. In addition, we also implement SAM that can generate masks for all objects in whole image.


We provide the pretrained model parameters of PaddlePaddle format, including [vit_b](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams), [vit_l](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams), [vit_h](https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams) and [vit_t](https://paddleseg.bj.bcebos.com/dygraph/paddlesegAnything/vit_t/model.pdparam) for [MobileSAM](https://github.com/ChaoningZhang/MobileSAM). For text prompt, we also provide the [CLIP_ViT_B](https://bj.bcebos.com/paddleseg/dygraph/clip/vit_b_32_pretrain/clip_vit_b_32.pdparams) model parameters of PaddlePaddle format.

## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance

<div align="center">
<img src="https://user-images.githubusercontent.com/18344247/232466911-f8d1c016-2eb2-46aa-94e2-3ec435f38502.gif"  width="1000" />
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
    pip install ftfy regex
    cd contrib/SegmentAnything/
    ```
* Download the example image to ```contrib/SegmentAnything/examples``` and the vocab to ```contrib/SegmentAnything/```
    ```bash
    wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
    wget https://bj.bcebos.com/paddleseg/dygraph/bpe_vocab_16e6/bpe_simple_vocab_16e6.txt.gz
    ```
    Then, the file structure is as following:

    ```
    PaddleSeg/contrib
    ├── SegmentAnything
    │   ├── examples
    │   │   └──  cityscapes_demo.png
    │   ├── segment_anything
    │   ├── scripts
    │   └── bpe_simple_vocab_16e6.txt.gz

    ```
### 2. Segment Anything on webpage.

In this step, we start a gradio service with the following scrip on local machine and you can try out our project with your own images.
Based on this service, You can experience the ability to **segment the whole image** and **segment the object based on text prompts**.

1. Run the following script:
    ```bash
    python scripts/text_to_sam_clip.py --model-type [vit_l/vit_b/vit_h/vit_t] # default is vit_h
    ```
    Note:
    *  There are three SAM model options for you, `vit_b`, `vit_l`, `vit_h`and `vit_t`, represent vit_base, vit_large, vit_huge and vit_mobilesam. Large model is more accurate but slower. You can choose the suitable model size based on your device.
    * We support `CLIP Vit-B` model for extracting text and image features.
    * `SAM vit_h` needs 16G memory and costs around 10s to infer an image on V100.

2. Open the webpage on your localhost: ```http://0.0.0.0:8078```
3. Try it out by clear and upload the test image! Our example looks like:

    <div align="center">
    <img src="https://user-images.githubusercontent.com/18344247/232427677-a7f913df-4abf-46ce-be2c-e37cbd495105.png"  width = "1000" />  
    </div>


### 3. Segment the object with point or box prompts

You can run the following commands to produce masks from different types of prompts including points and boxes, as follow. The picture result will be saved to the `output/` directory, with the image name the same as the input.


1. Box prompt

```bash
python scripts/promt_predict.py --input_path xxx.png --box_prompt 1050 370 1500 700 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

2. Point prompt
```bash
python scripts/promt_predict.py --input_path xxx.png --point_prompt 1200 450 --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

### 4. Segment the object only with whole image

```bash
python scripts/amg_paddle.py --model-type [vit_l/vit_b/vit_h/vit_t] --model-type [vit_l/vit_b/vit_h] # default is vit_h
```

## Reference

> Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick. [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/).

> Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8748-8763, 2021. [CLIP](https://github.com/openai/CLIP)
