# Web 端部署

## 1.介绍
以人像分割在 MacOS Chrome 的部署为例，介绍如何使用前端推理引擎 [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) 对分割模型进行部署。文档第二部分介绍如何使用人像分割模型 js 库 [@paddlejs-models/humanseg](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-models/humanseg)，第三部分介绍重点 API。

## 2. 使用

### 2.1 要求
* 安装 Node （https://nodejs.org/zh-cn/download/）
* 确认是否安装成功，在命令行执行
```sh
# 显示所安 node 版本号，即表示成功安装
node -v
```
### 2.2 步骤
```sh
# clone Paddle.js
git clone https://github.com/PaddlePaddle/PaddleSeg.git

# 进入 deploy web example 目录，安装依赖
cd PaddleSeg/deploy/web/example/ && npm install

# 执行命令
npm run dev

# 访问 http://0.0.0.0:8866/ ，即可体验人像分割处理图片应用
```


### 2.3 效果展示

![image](https://user-images.githubusercontent.com/10822846/118273079-127bf480-b4f6-11eb-84c0-8a0bbc7c7433.png)

## 3. 重点 API 介绍

## 3.1 @paddlejs-models/humanseg 介绍
npm 库 [@paddlejs-models/humanseg](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-models/humanseg) 封装了前端推理引擎 [Paddle.js](https://github.com/PaddlePaddle/Paddle.js) 和计算方案 [paddlejs-backend-webgl](https://github.com/PaddlePaddle/Paddle.js/tree/master/packages/paddlejs-backend-webgl)，该计算方案通过 WebGL 获得 GPU 加速。
所以只需引入库 @paddlejs-models/humanseg 即可，无需再额外引入推理引擎和计算方案。

## 3.1.1 API 介绍
@paddlejs-models/humanseg 暴露了四个 API：
* load
调用 load API 完成推理引擎初始化。下载humanseg_lite_inference web 模型，根据模型结构和参数文件生成神经网络，并完成模型预热。

* getGrayValue
给 getGrayValue API 传入需要处理的图片，执行后获得推理结果。

* drawHumanSeg
绘制人像。给 drawHumanSeg API 传入 canvas 画布元素和推理结果，可以通过该元素传递背景信息，分割后的人像将绘制在此 canvas上。

* drawMask
绘制人像遮罩。传入 canvas 画布元素和推理结果，同时传入参数 dark 配置是否使用暗黑模式。效果将绘制在传入的 canvas 画布元素上。

## 3.2 如何使用 @paddlejs-models/humanseg

```js
// 引入
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
