
# 从训练到推理部署工具链测试方法介绍

test_model.sh和params.txt文件配合使用，完成分割模型从训练到预测的流程测试。

# 安装依赖
- 安装PaddlePaddle >= 2.1.2
- 安装PaddleSeg依赖
    ```
    pip3 install  -r ../requirements.txt
    ```
- 安装autolog
    ```
    git clone https://github.com/LDOUBLEV/AutoLog
    cd AutoLog
    pip3 install -r requirements.txt
    python3 setup.py bdist_wheel
    pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
    cd ../
    ```

# 目录介绍

```bash
tests/
├── hrnetw18_small_v1_params.txt                                  # 测试分割模型的参数配置文件
└── fcn_hrnetw18_small_v1_humanseg_192x192_mini_supervisely.yml   # 测试分割模型的config配置文件
└── prepare.sh                                                    # 完成test_model.sh运行所需要的数据和模型下载
└── test_model.sh                                                 # 测试主程序
```

# 使用方法

test_model.sh包含四种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test/prepare.sh ./tests/hrnetw18_small_v1_params.txt 'lite_train_infer'
bash tests/test_model.sh ./tests/hrnetw18_small_v1_params.txt 'lite_train_infer'
```  

- 模式2：whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash tests/prepare.sh ./tests/hrnetw18_small_v1_params.txt 'whole_infer'
bash tests/test_model.sh ./tests/hrnetw18_small_v1_params.txt 'whole_infer'
```  

- 模式3：infer 不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash tests/prepare.sh ./tests/hrnetw18_small_v1_params.txt 'infer'
# 用法1:
bash tests/test_model.sh ./tests/hrnetw18_small_v1_params.txt 'infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash tests/test_model.sh ./tests/hrnetw18_small_v1_params.txt 'infer' '1'
```  

- 模式4：whole_train_infer , CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash tests/prepare.sh ./tests/hrnetw18_small_v1_params.txt 'whole_train_infer'
bash tests/test_model.sh ./tests/hrnetw18_small_v1_params.txt 'whole_train_infer'
```  


# 日志输出
最终在```tests/output```目录下生成.log后缀的日志文件
