# Segment Anything with PaddleSeg

## Contents
1. Overview
2. News
2. Performance
3. Try it by yourself with one line of code

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> Overview


## <img src="https://user-images.githubusercontent.com/34859558/190044217-8f6befc2-7f20-473d-b356-148e06265205.png" width="25"/> Performance


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
In this step, we will start a gradio service on local machine and you can try out our project with your own images.

There are three model options for you, vit_b, vit_l and vit_h, represent vit_base, vit_large and vit_huge. Large model is more accurate and also slower. You can choose the model size based on your device. The test result shows that vit_h needs 16G video memory and needs around 10s to infer an image on V100.

```bash
python script/amg_paddle.py --model-type [vit_l/vit_b/vit_h] # default is vit_h

```
