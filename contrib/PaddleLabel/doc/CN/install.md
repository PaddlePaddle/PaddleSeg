# 安装说明

为避免环境问题，建议首先创建一个新的虚拟环境

```python
conda create -n paddlelabel python=3.10
conda activate paddlelabel
```

您可以通过以下三种方式中**任意一种**安装PaddleLabel，其中通过 pip 安装最简单。

## 通过 pip 安装

```shell
pip install --upgrade paddlelabel
```

看到类似于 `Successfully installed paddlelabel-0.5.0` 的命令行输出即为安装成功，您可以直接继续浏览[启动](#%E5%90%AF%E5%8A%A8)章节。

以下两种安装方式主要针对开发者。

## 下载最新开发版

<details>
<summary>详细步骤</summary>
<br>
PaddleLabel 团队会不定期在项目更新后通过 Github Action 构建反映最新版代码的安装包。这一安装包未经过全面测试，可能包含一些问题，仅推荐为尝试最新版本使用。其中可能修复了一些 pypi 版本中存在的问题和进行了一些性能提升。

下载方式为

1. 访问 [Action 执行记录网页](https://github.com/PaddleCV-SIG/PaddleLabel/actions/workflows/pypi.yml)
1. 选择最上面（最新）的一次执行，点击进入

![1](https://user-images.githubusercontent.com/29757093/201906327-18444fcb-57b7-4e5f-8e00-62bf1e3b49b7.png)

3. 下载 PaddleLabel_built_package

![1](https://user-images.githubusercontent.com/29757093/201905747-a2b0901c-9331-4a90-b4ae-44c855314810.jpg)

4. 解压该压缩包，之后执行

```shell
pip install [解压出的.whl文件名，如 paddlelabel-0.2.0-py3-none-any.whl ]
```

</details>

## 通过源码安装

<details>
<summary>详细步骤</summary>
<br>

1. 首先需要将后端代码（本项目）克隆到本地

```shell
git clone https://github.com/PaddleCV-SIG/PaddleLabel
```

2. 接下来需要克隆并构建前端，需要首先安装 [Node.js](https://nodejs.org/en/) 和 npm

```shell
git clone https://github.com/PaddleCV-SIG/PaddleLabel-Frontend
cd PaddleLabel-Frontend
npm install --location=global yarn
yarn
npm run build
```

3. 将构建好的前端部分，`PaddleLabel-Frontend/dist/`目录下所有文件复制到`paddlelabel/static/`中

```shell
cd ../PaddleLabel/
mkdir paddlelabel/static/
cp -r ../PaddleLabel-Frontend/dist/* paddlelabel/static/
```

4. 安装PaddleLabel

```shell
# 在PaddleLabel目录下
python setup.py install
```

</details>

# 启动

完成上述的安装操作后，可以直接在终端使用如下指令启动 PaddleLabel

```shell
paddlelabel  # 启动paddlelabel
pdlabel # 缩写，和paddlelabel完全相同
```

PaddleLabel 默认运行在[http://127.0.0.1:17995](http://127.0.0.1:17995)上，可以通过`--port`或`-p`参数指定端口。此外可以通过`--lan`或`-l`参数将服务暴露到局域网。这样可以实现在电脑上运行 PaddleLabel，在平板上进行标注。在 docker 中运行 PaddleLabel 时也需要添加`--lan`参数。

```shell
paddlelabel --port 8000 --lan  # 在8000端口上运行并将服务暴露到局域网
```

启动后PaddleLabel会自动在浏览器种打开网页，推荐使用Chrome。
