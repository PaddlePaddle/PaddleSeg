English | [简体中文](install_cn.md)


## Environment Requirements

- PaddlePaddle 2.2
- OS: 64-bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64-bit version
- pip/pip3(9.0.1+)，64-bit version 
- CUDA >= 10.1 
- cuDNN >= 7.6 


### 1. Install PaddlePaddle

Highly recommend you install the GPU version of PaddlePaddle, due to the large overhead of segmentation models, otherwise, it could be out of memory while running the models.

```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.2.1.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CPU
python -m pip install paddlepaddle==2.2.1 -i https://mirror.baidu.com/pypi/simple
```
- For quick installation of more CUDA versions or environments, please refer to [PaddlePaddle Quick Installation Document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or source code compilation and installation methods, please refer to [PaddlePaddle Installation Document](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# Confirm that PaddlePaddle is installed successfully in your Python interpreter
>>> import paddle
>>> paddle.utils.run_check()

# Confirm PaddlePaddle version
python -c "import paddle; print(paddle.__version__)"

# If the following prompt appears on the command line, the PaddlePaddle installation is successful.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```



## 2. Install PaddleSeg
Support API method for flexible development.
```
pip install paddleseg
```

## 3. Download Repo
Support Configuration Drive for simple and fast development.

```
git clone https://github.com/PaddlePaddle/PaddleSeg
```

## 4. Install Dependencies
```
cd PaddleSeg
pip install -r requirements.txt

#If a version error occurs during installation, you can try to delete the old version and re-run the script.
```
## 5. Verify Installation

Run the following command to verify PaddleSeg installation. The predicted results will be in output/result if successful.

```python
python predict.py \
       --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
       --model_path https://bj.bcebos.com/paddleseg/dygraph/optic_disc/bisenet_optic_disc_512x512_1k/model.pdparams\
       --image_path docs/images/optic_test_image.jpg \
       --save_dir output/result
```
