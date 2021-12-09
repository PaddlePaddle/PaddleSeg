# Web 端基础预测功能测试

Web 端主要基于 Jest-Puppeteer 完成 e2e 测试，其中 Puppeteer 操作 Chrome 完成推理流程，Jest 完成测试流程。
>Puppeteer 是一个 Node 库，它提供了一个高级 API 来通过 DevTools 协议控制 Chromium 或 Chrome
>Jest 是一个 JavaScript 测试框架，旨在确保任何 JavaScript 代码的正确性。

#### 环境准备

* 安装 Node（包含 npm ） （https://nodejs.org/zh-cn/download/）
* 确认是否安装成功，在命令行执行
```sh
# 显示所安 node 版本号，即表示成功安装
node -v
```
* 确认 npm 是否安装成成
```sh
# npm 随着 node 一起安装，一般无需额外安装
# 显示所安 npm 版本号，即表示成功安装
npm -v
```

#### 使用
```sh
# web 测试环境准备
bash test_tipc/prepare_js.sh 'js_infer'

# web 推理测试
bash test_tipc/test_infer_js.sh
```


#### 流程设计

###### paddlejs prepare
 1. 判断 node, npm 是否安装
 2. 下载测试模型，当前为 ppseg_lite_portrait_398x224 ，如果需要替换，把对应模型连接和模型包名字修改即可
  - 当前模型：https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz
  - 模型配置：configs/pp_humanseg_lite/pp_humanseg_lite_export_398x224.yml
 3. 导出模型为静态图模型（保存地址：test_tipc/web/models/pphumanseg_lite）
 4. 转换模型， model.pdmodel model.pdiparams 转换为 model.json chunk.dat
 5. 安装最新版本 humanseg sdk  @paddlejs-models/humanseg@latest
 6. 安装测试环境依赖 puppeteer、jest、jest-puppeteer，如果检查到已经安装，则不会进行二次安装


 ###### paddlejs infer test
 1. Jest 执行 server command：`python3 -m http.server 9811` 开启本地服务
 2. 启动 Jest 测试服务，通过 jest-puppeteer 插件完成 chrome 操作，加载 @paddlejs-models/humanseg 脚本完成推理流程
 3. 测试用例为分割后的图片与预期图片（expect img）效果进行**逐像素对比**，精度误差不超过 **2%**，则视为测试通过，通过为如下显示：
 <img width="594" alt="infoflow 2021-12-02 16-34-05" src="https://user-images.githubusercontent.com/
 10822846/144386307-b4e10b07-f105-499f-b953-4dc2707c6242.png">
