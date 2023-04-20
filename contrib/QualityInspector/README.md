# 工业质检全流程解决方案  **QualityInspector**

## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="25"/> 简介

在3C电子、汽车、纺织化纤、金属、建筑、食品、日用消费品等生产制造行业，质量检测是保障产品质量的重要一环，是企业确保产品符合标准、满足客户需求、提高竞争力的关键步骤。在深度学习时代，AI赋能工业质检成为大势所趋。传统的数字图像处理方法虽然比起人工检测已经有了很大进步，但往往存在精度不高、泛化性差等问题。基于深度学习的方法能够很大程度上缓解这些问题，给很多复杂的质检场景带来自动化的可能性。

基于飞桨的计算机视觉开发套件PaddleClas、PaddleSeg、PaddleDetection三件套，已经可以解决很多质检问题，但是这些套件没有提供针对工业质检场景的数据预处理、后处理、评测指标等配套工具，没有提供针对工业质检的特色模型，且不能有效支持需要这些套件联动来解决问题的场景。

**QualityInspector工业质检全流程解决方案开发工具，致力于帮助开发者快速完成算法的研发、验证和调优，端到端完成从数据标注到模型部署的全流程工业质检应用。**

QualityInspector目前发布V0.5预览版本，主要特性包括：
* 统一可配置的解决方案：支持检测、分割单模型、检测+RoI分割串联结合后处理的解决方案，简单修改配置即可轻松组合视觉套件的模型。
* 工业级指标评估和调优：评估工业质检项目实际落地指标，并可直接调节后处理规则参数进行指标一键调优，方便易用。
* 丰富的视觉算法库：新增支持无监督异常检测算法，同时集成飞桨视觉套件的成熟算法库，覆盖图像分割、目标检测等任务。
* 可快速上手的工具：支持数据格式转化工具，快速完成检测，分割/RoI分割任务数据格式转化，同时支持数据分析工具和EISeg数据标注工具。


<div align="center">
<img src="https://github.com/Sunting78/images/blob/master/all0.png"  width="900" />
</div>

QualityInspector部分可视化效果如下：

<div align="center">
<img src="https://github.com/Sunting78/images/blob/master/ezgif.com-video-to-gif.gif"  width="900" />
</div>


## <img src="https://user-images.githubusercontent.com/34859558/190043516-eed25535-10e8-4853-8601-6bcf7ff58197.png" width="25"/> 最新消息

* [2023-4] 🔥 **发布全流程工业质检开发工具QualityInspector V0.5版本**
  * 支持配置化全流程方案搭建，简单修改配置即可轻松组合视觉套件的模型。
  * 支持无监督异常检测算法。
  * 提供工业质检场景使用的数据通用工具脚本。
  * 支持全流程工业指标评测，后处理调优。


## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> 详细教程

### 1. [安装说明](./docs/install.md)
### 2. [快速开始](./docs/quick_start.md)
### 3. 数据准备
   * [准备数据集](./docs/tools_data/prepare_data.md)
   * [数据集格式转换工具](./docs/tools_data/conver_tools.md)
   * [数据分析工具](./docs/tools_data/parse_tools.md)
   * [EISeg 数据标注](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/EISeg)

### 4. 训练&验证
   * [检测分割算法](./docs/det_seg/train_eval.md)
   * [无监督异常检测算法](./docs/uad/README.md)

### 5. 全流程预测
   * [准备全流程配置文件](./docs/end2end/parse_config.md)
   * [全流程预测](./docs/end2end/predict.md)

### 6. 全流程评估
   * [过杀&漏检指标评估](./docs/end2end/eval.md#全流程评估)
   * [badcase可视化分析](./docs/end2end/eval.md#badcase可视化输出)
   * [后处理参数调优](./docs/end2end/eval.md#后处理参数调整)

## <img src="../../docs/images/chat.png" width="25"/> 技术交流
* 如果大家有使用问题和功能建议, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
* **欢迎加入PaddleSeg的微信用户群👫**（扫码填写简单问卷即可入群），大家可以**领取30G重磅学习大礼包🎁**，也可以和值班同学、各界大佬直接进行交流。
  * 🔥 获取深度学习视频教程、图像分割论文合集
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等

<div align="center">
<img src="https://user-images.githubusercontent.com/30883834/213601179-0813a896-11e1-4514-b612-d145e068ba86.jpeg"  width = "200" />  
</div>


## <img src="https://user-images.githubusercontent.com/34859558/190046674-53e22678-7345-4bf1-ac0c-0cc99718b3dd.png" width="25"/> TODO
未来，我们想在这几个方面来发展 QualityInspector，欢迎加入我们的开发者小组。
- [] 图像配准对齐。
- [] 小目标，长尾分布，少样本等问题的研究。
- [] ...


## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="25"/> License

QualityInspector 的 License 为 [Apache 2.0 license](LICENSE).

## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="25"/> 致谢
