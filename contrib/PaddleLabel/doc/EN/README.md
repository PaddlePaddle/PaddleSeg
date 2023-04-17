# PaddleLabel

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) ![PyPI](https://img.shields.io/pypi/v/paddlelabel?color=blue) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Start](https://img.shields.io/github/stars/PaddleCV-SIG/PaddleLabel?color=orange)](<>) [![Fork](https://img.shields.io/github/forks/PaddleCV-SIG/PaddleLabel?color=orange)](<>) ![PyPI - Downloads](https://img.shields.io/pypi/dm/paddlelabel?color=orange) [![OS](https://img.shields.io/badge/os-linux%2C%20windows%2C%20macos-green.svg)](<>)

<!-- <a href=""><img src="https://img.shields.io/badge/QQ_Group-1234567-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20"></a> -->

PaddleLabel aims to become an effective and flexible data annotation tool. There are three parts to this project. This repo contains backend implementation. [PaddleLabel-Frontend](https://github.com/PaddleCV-SIG/PaddleLabel-Frontend) contains the React/Antd frontend. [PaddleLabel-ML](https://github.com/PaddleCV-SIG/PaddleLabel-ML) contains the machine learning backend for automatic and interactive models.

## Install

Installing in a new environment is not required but suggested.

```python
conda create -n pplabel python=3.9
conda activate pplabel
```

### pip

```shell
pip install paddlelabel
paddlelabel
```

paddlelabel is now running at [http://127.0.0.1:17995](http://127.0.0.1:17995)

### source

First clone this repo for backend code.

```shell
git clone https://github.com/PaddleCV-SIG/PaddleLabel
```

Then clone and build frontend

```shell
git clone https://github.com/PaddleCV-SIG/PaddleLabel-Frontend
cd PaddleLabel-Frontend
npm install -g yarn
yarn
npm run build
cd ..
```

The last step is to copy built frontend to

```shell
cd PaddleLabel
pip install -r requirements.txt
mkdir paddlelabel/static/
cp -r ../PaddleLabel-Frontend/dist/* paddlelabel/static/

python setup.py install
```

## Run

After installation, run PaddleLabel from command line with

```shell
paddlelabel
```

PaddleLabel is now avaliable at [http://127.0.0.1:17995](http://127.0.0.1:17995)

You can also choose to expose the service to lan. This way it's possbile to run the service on a computer and annotate with a tablet.

```shell
paddlelabel --lan
```

## Dataset Import/Export

PaddleLabel currently support image classification, object detection and image segmentation projects. Please refer to the [Dataset File Structure Documentation](project/dataset_file_structure.md) for more details.

## Release Notes

- 2022.5.31: v0.1.0 \[1\] Support image classification, detection and segmentations. \[2\] Interactive image segmentation with EISeg models

## Contribute

Please refer to the [Developers Guide](developers_guide.md) for details on backend implementation.
