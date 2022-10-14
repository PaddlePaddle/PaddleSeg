English | [简体中文](install.md)

## Installation

EISeg provides multiple ways of installation, among which [pip](#PIP) and [run code](#run code) are compatible with Windows, Mac OS and Linux. It is recommended to install in a virtual environment created by conda for fear of environmental conflicts.

Version Requirements:

- PaddlePaddle >= 2.2.0

For more details of the installation of PaddlePaddle, please refer to our [official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html).

### Clone

Clone PaddleSeg to your local system through git:

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

Install the required environment (if you need to use GDAL and SimpleITK, please refer to **Vertical Segmentation** for installation).

```
pip install -r requirements.txt
```

Enable EISeg by running eiseg after installing the needed environment:

```
cd PaddleSeg\EISeg
python -m eiseg
```

Or you can run exe.py in eiseg:

```
cd PaddleSeg\EISeg\eiseg
python exe.py
```

### PIP

Install pip as follows：

```
pip install eiseg
```

pip will install dependencies automatically. After that, enter the following at the command line:

```
eiseg
```

Now, you can run pip.
