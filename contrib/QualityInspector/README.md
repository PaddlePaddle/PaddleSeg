# 全流程工业质检开发工具 **QualityInspector**

## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> 简介

在3C电子、汽车、纺织化纤、金属、建筑、食品、日用消费品等生产制造行业，质量检测是保障产品质量的重要一环，是企业确保产品符合标准、满足客户需求、提高竞争力的关键步骤。在深度学习时代，AI赋能工业质检成为大势所趋。传统的数字图像处理方法虽然比起人工检测已经有了很大进步，但往往存在精度不高、泛化性差等问题。基于深度学习的方法能够很大程度上缓解这些问题，给很多复杂的质检场景带来自动化的可能性。

基于飞桨的计算机视觉开发套件PaddleClass、PaddleSeg、PaddleDetection三件套，已经可以解决很多质检问题，但是这些套件没有提供针对工业质检场景的数据预处理、后处理、评测指标等配套工具，没有提供针对工业质检的特色模型，且不能有效支持需要这些套件联动来解决问题的场景。

**QualityInspector全流程工业质检开发工具致力于帮助企业开发者快速完成算法的研发、验证和调优，端到端完成从数据标注到模型部署的全流程工业质检应用**

QualityInspect目前发布V0.5预览版本，主要特性包括：
* 统一可配置的解决方案，支持检测/分割单模型，或检测+RoI分割串联的解决方案。
* 丰富的视觉模型库：集成飞桨视觉套件的成熟模型库，覆盖图像分割、目标检测、场景分类等任务。
* 可快速上手的案例：基于公开数据的解决方案的方法评测，帮助用户使用、分析和选择pipeline。
* 此外，还包括针对工业质检领域的特色支持：
   * 数据层面：支持数据格式转化工具，快速完成检测，分割/RoI分割任务数据格式转化和数据可视化和类别统计等工具。
   * 后处理模块：针对得分，长度，面积等可配置的参数，降低过杀。
   * 评测分析工具：工业项目指标，badcase分析，后处理参数调优。
   * 冷启动：集成无监督异常检测算法。

## <img src="https://user-images.githubusercontent.com/34859558/190043516-eed25535-10e8-4853-8601-6bcf7ff58197.png" width="25"/> 最新消息

* [2023-4] **发布全流程工业质检开发工具QualityInspector V0.5版本**
  * 支持无监督异常检测算法。
  * 支持配置化pipeline搭建，简单修改配置即可轻松组合视觉套件的模型。
  * 提供工业质检场景使用的数据通用工具脚本。
  * 支持全流程工业指标评测，后处理调优。


## <img src="https://user-images.githubusercontent.com/34859558/190043857-bfbdaf8b-d2dc-4fff-81c7-e0aac50851f9.png" width="25"/> 快速开始

### 1. [安装说明](./docs/install.md)
### 2. 数据准备
   * [准备数据集](./docs/tools_data/prepare_data.md)
   * [数据集格式转换工具](./docs/tools_data/conver_tools.md)
   * [数据分析工具](./docs/tools_data/parse_tools.md)
   * [EISeg 数据标注](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/EISeg)

### 3. 训练/推理
   * [无监督异常检测算法](./docs/uad/README.md)
   * [检测分割算法](./docs/det_seg/train_eval.md)

### 4. 全流程预测
   * [准备全流程配置文件](./docs/end2end/parse_config.md)
   * [全流程预测](./docs/end2end/predict.md)

### 5. 全流程评估（过杀/漏检）
   * [指标评估](./docs/end2end/eval.md)
   * [badcase可视化分析](./docs/end2end/eval.md)
   * [后处理参数调优](./docs/end2end/eval.md)

## <img src="../../docs/images/chat.png" width="25"/> 技术交流
* 如果大家有使用问题和功能建议, 可以通过[GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues)提issue。
* **欢迎加入PaddleSeg的微信用户群👫**（扫码填写简单问卷即可入群），大家可以**领取30G重磅学习大礼包🎁**，也可以和值班同学、各界大佬直接进行交流。
  * 🔥 获取深度学习视频教程、图像分割论文合集
  * 🔥 获取PaddleSeg的历次直播视频，最新发版信息和直播动态
  * 🔥 获取PaddleSeg自建的人像分割数据集，整理的开源数据集
  * 🔥 获取PaddleSeg在垂类场景的预训练模型和应用合集，涵盖人像分割、交互式分割等等
  * 🔥 获取PaddleSeg的全流程产业实操范例，包括质检缺陷分割、抠图Matting、道路分割等等
<div align="center">
<img src="https://user-images.githubusercontent.com/48433081/174770518-e6b5319b-336f-45d9-9817-da12b1961fb1.jpg"  width = "200" />  
</div>


## <img src="https://user-images.githubusercontent.com/34859558/190046674-53e22678-7345-4bf1-ac0c-0cc99718b3dd.png" width="25"/> TODO
未来，我们想在这几个方面来发展 QualityInspector，欢迎加入我们的开发者小组。
- [✔️] 图像配准对齐。
- [✔️] 小目标，长尾分布，少样本等问题的研究。
- [✔️] ...


## <img src="https://user-images.githubusercontent.com/34859558/188446853-6e32659e-8939-4e65-9282-68909a38edd7.png" width="25"/> License

QualityInspector 的 License 为 [Apache 2.0 license](LICENSE).

## <img src="https://user-images.githubusercontent.com/34859558/188446803-06c54d50-f2aa-4a53-8e08-db2253df52fd.png" width="25"/> 致谢
