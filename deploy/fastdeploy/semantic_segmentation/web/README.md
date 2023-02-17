English | [简体中文](README_CN.md)
# PP-Humanseg v1 Model Frontend Deployment

## Model Version

- [PP-HumanSeg Release/2.6](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/contrib/PP-HumanSeg/)


## Deploy PP-Humanseg v1 Model on Frontend

To deploy and use PP-Humanseg v1 model of web demo, please refer to [document](../../../../application/js/web_demo/README.md).


## PP-Humanseg v1 js interface

```
import * as humanSeg from "@paddle-js-models/humanseg";
# Load and initialise model
await humanSeg.load(Config);
# Portrait segmentation
const res = humanSeg.getGrayValue(input)
# Extract the binary map of portrait and background
humanSeg.drawMask(res)
# Visualization function for background replacement
humanSeg.drawHumanSeg(res)
# Blur background
humanSeg.blurBackground(res)
```

**Parameters in function load()**
> * **Config**(dict): Configuration parameter for PP-Humanseg model, default is {modelpath : 'https://paddlejs.bj.bcebos.com/models/fuse/humanseg/humanseg_398x224_fuse_activation/model.json', mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], enableLightModel: false}；modelPath is the default PP-Humanseg js model. Mean, std respectively represent the mean and standard deviation of the preprocessing, and enableLightModel represents whether to use a lighter model.


**Parameters in function getGrayValue()**
> * **input**(HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Input image parameter.

**Parameters in function drawMask()**
> * **seg_values**(number[]): Input parameter, generally the result of function getGrayValue is used as input.

**Parameters in function blurBackground()**
> * **seg_values**(number[]): Input parameter, generally the result of function getGrayValue is used as input.

**Parameters in function drawHumanSeg()**
> * **seg_values**(number[]): Input parameter, generally the result of function getGrayValue is used as input.
