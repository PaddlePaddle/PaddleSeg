English | [简体中文](README_cn.md)

# Web deployment

## 1 Introduction
Taking the deployment of portrait segmentation on MacOS Chrome as an example, this paper introduces how to use the front-end inference engine [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) to deploy the segmentation model. The second part of the document describes how to use the portrait segmentation model js library [@paddlejs-models/humanseg](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-models/humanseg), the third Section introduces key APIs.

## 2. Use

### 2.1 Requirements
* Install Node (https://nodejs.org/zh-cn/download/)
* Confirm whether the installation is successful, execute it on the command line
````sh
# Display the installed node version number, which means successful installation
node -v
````
### 2.2 Steps
````sh
# clone Paddle.js
git clone https://github.com/PaddlePaddle/PaddleSeg.git

# Enter the deploy web example directory and install dependencies
cd PaddleSeg/deploy/web/example/ && npm install

# Execute
npm run dev

# Visit http://0.0.0.0:8866/ to experience the application of portrait segmentation and processing
````


### 2.3 Effect display

![image](https://user-images.githubusercontent.com/10822846/118273079-127bf480-b4f6-11eb-84c0-8a0bbc7c7433.png)

## 3. Key API Introduction

## 3.1 Introduction @paddlejs-models/humanseg
The npm library [@paddlejs-models/humanseg](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-models/humanseg) encapsulates the front-end inference engine [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) and calculation scheme [paddlejs-backend-webgl](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-backend-webgl), The computing scheme is GPU-accelerated through WebGL.
So just import the library @paddlejs-models/humanseg, no need to introduce additional inference engine and calculation scheme.

## 3.1.1 API Introduction
@paddlejs-models/humanseg exposes four APIs:
* load
Call the load API to complete the initialization of the inference engine. Download the humanseg_lite_inference web model, generate the neural network according to the model structure and parameter file, and complete the model warm-up.

* getGrayValue
Pass the image to be processed to the getGrayValue API, and get the inference result after execution.

* drawHumanSeg
Draw portraits. Pass the canvas element and the inference result to the drawHumanSeg API, you can pass the background information through this element, and the segmented portrait will be drawn on this canvas.

* drawMask
Draw a portrait mask. Pass in the canvas element and inference result, and pass in the parameter dark to configure whether to use dark mode. The effect will be drawn on the incoming canvas element.

## 3.2 How to use @paddlejs-models/humanseg

```js
// import
import * as humanseg from '@paddlejs-models/humanseg';

// load humanseg model
await humanseg.load();

// get the seg value [192 * 192];
const { data } = await humanseg.getGrayValue(img);

// draw human segmentation
const canvas1 = document.getElementById('demo1') as HTMLCanvasElement;
humanseg.drawHumanSeg(canvas1, data);

// draw the background mask
const canvas2 = document.getElementById('demo2') as HTMLCanvasElement;
humanseg.drawMask(canvas2, data, true);

```
