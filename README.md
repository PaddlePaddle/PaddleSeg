>> 最好模型地址：链接：https://pan.baidu.com/s/1gPuqmhG6K2vo2eyBchEuqg 提取码：CAVS   
>> 提取后放到PaddleSeg/output目录下

>> 预训练模型地址：链接：https://pan.baidu.com/s/1zDcUb4GxHH1xweRu3l2M9A 提取码：CAVS    
>> 预训练模型保存在PaddleSeg/pretrained中    

## 第一步克隆本项目：  
~~~shell
git clone https://github.com/marshall-dteach/PSA.git
cd PaddleSeg
~~~
## 第二步数据集格式
数据集格式    
|-cityscapes    
>|-leftImg8bit     
>>|-train     
>>|-val  

>|-gtFine    
>>|-train    
>>|-val   
 ## 第三步训练模型和模型预测
  #训练模型
  ~~~shell
  cd PaddleSeg
  python main.py #单卡
  python -m paddle.distributed.launch main.py  #多卡
  ~~~
  #验证模型
  ~~~shell
  cd PaddleSeg
  python valide.py
  ~~~
### 注意优化器选取根据单卡和四卡不同而调整
![image](https://user-images.githubusercontent.com/63546191/169228623-87f12422-54a4-449d-b42d-7e4d9baa22b0.png)
注意：单卡训练使用上面的lr，四卡训练使用下面的lr    

      
   
## 第四步模型精度检验
![image](https://user-images.githubusercontent.com/63546191/169260254-0892fcf1-1408-4bbf-9049-6dd7a3a5a88b.png)   
精度要求miou达到86.7，我的模型精度为86.87，满足要求

## 动态模型验证     
比较模型预测与ground truth    
![1](https://user-images.githubusercontent.com/63546191/169755335-068bbf51-25c2-4bc3-a589-adcc5c2261eb.png)  
![2](https://user-images.githubusercontent.com/63546191/169755356-e49bd5d2-b293-467f-8822-c40e959536e7.png)    
![3](https://user-images.githubusercontent.com/63546191/169755371-fe093a13-7115-4b86-9faf-1104c8c4c8b0.png)   
![4](https://user-images.githubusercontent.com/63546191/169755407-3fb01395-ec1d-4398-bfc8-20d42ce3950b.png)      
![5](https://user-images.githubusercontent.com/63546191/169755436-936867a7-d53f-4588-9b48-72fff455dc70.png)     
![6](https://user-images.githubusercontent.com/63546191/169755571-93992eb7-2a6e-4e3f-aa5f-10105d45f505.png)   

## TIPC问题    
>第一步：加载模型

~~~Python
%cd PaddleSeg
from paddleseg.models import MscaleOCR
import paddle
model = MscaleOCR(num_classes=19)
state_dict = paddle.load('model.pdparams')
model.set_state_dict(state_dict)
~~~

> 第二步：export模型，命令如下

~~~shell
!python export.py --config=configs/Mscale/model.yml --model_path=model.pdparams --input_shape 1 3 1024 2048
~~~

其中export.py是PaddleSeg自带的，内容如下

~~~Python
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import yaml

from paddleseg.cvlibs import Config
from paddleseg.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        '--without_argmax',
        dest='without_argmax',
        help='Do not add the argmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        '--with_softmax',
        dest='with_softmax',
        help='Add the softmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


class SavedSegmentationNet(paddle.nn.Layer):
    def __init__(self, net, without_argmax=False, with_softmax=False):
        super().__init__()
        self.net = net
        self.post_processer = PostPorcesser(without_argmax, with_softmax)

    def forward(self, x):
        outs = self.net(x)
        outs = self.post_processer(outs)
        return outs


class PostPorcesser(paddle.nn.Layer):
    def __init__(self, without_argmax, with_softmax):
        super().__init__()
        self.without_argmax = without_argmax
        self.with_softmax = with_softmax

    def forward(self, outs):
        new_outs = []
        for out in outs:
            if self.with_softmax:
                out = paddle.nn.functional.softmax(out, axis=1)
            if not self.without_argmax:
                out = paddle.argmax(out, axis=1)
            new_outs.append(out)
        return new_outs


def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    cfg = Config(args.cfg)
    net = cfg.model

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape

    if not args.without_argmax or args.with_softmax:
        new_net = SavedSegmentationNet(net, args.without_argmax,
                                       args.with_softmax)
    else:
        new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(
            shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        transforms = cfg.export_config.get('transforms', [{
            'type': 'Normalize'
        }])
        data = {
            'Deploy': {
                'transforms': transforms,
                'model': 'model.pdmodel',
                'params': 'model.pdiparams'
            }
        }
        yaml.dump(data, file)

    logger.info(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

~~~



下面为model.yml文件内容

~~~yaml
batch_size: 1
iters: 10

model:
  type: MscaleOCR
  num_classes: 19
 
~~~

运行结果如下

~~~shell
W0519 19:24:00.705386  1337 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0519 19:24:00.710723  1337 device_context.cc:465] device: 0, cuDNN Version: 7.6.
/home/aistudio/PaddleSeg/paddleseg/models/network/hrnetv2.py:320: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  self.high_level_ch = np.int(np.sum(pre_stage_channels))
2022-05-19 19:24:09 [INFO]	Loaded trained params of model successfully.
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return (isinstance(seq, collections.Sequence) and
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/dygraph_to_static/convert_call_func.py:93: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
  func_in_dict = func == v
2022-05-19 19:26:36 [INFO]	Model is saved in ./output.
~~~

运行完之后可以得到三个关于静态模型的文件和一个yaml文件，模型大小分别为：

|—deploy.yaml  91B

|— model.pdiparams  299.9MB

|— model.pdiparams.info 209.2KB

|— model.pdmodel 97MB



其中deploy.yaml文件的内容如下

~~~yaml
Deploy:
  model: model.pdmodel
  params: model.pdiparams
  transforms:
  - type: Normalize
~~~

> 第三步infer模型，其中infer.py为PaddleSeg自带的文件，代码如下

~~~shell
!python deploy/python/infer.py --image_path msmodels/demo.png --config=msmodels/deploy.yaml
~~~

infer.py代码如下

~~~Python
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import codecs
import os
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleseg.transforms as T
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.utils.visualize import get_pseudo_color_map


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")

    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to enable tuned dynamic shape. We uses some images to collect '
        'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default="auto_tune_tmp.pbtxt",
        help='The temp file to save tuned dynamic shape.')

    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')

    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = np.array([cfg.transforms(imgs[i])[0]])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i in range(0, len(imgs_path), args.batch_size):
            # warm up
            if i == 0 and args.benchmark:
                for j in range(5):
                    data = np.array([
                        self._preprocess(img)
                        for img in imgs_path[0:args.batch_size]
                    ])
                    input_handle.reshape(data.shape)
                    input_handle.copy_from_cpu(data)
                    self.predictor.run()
                    results = output_handle.copy_to_cpu()
                    results = self._postprocess(results)

            # inference
            if args.benchmark:
                self.autolog.times.start()

            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + args.batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            if args.benchmark:
                self.autolog.times.stamp()

            results = output_handle.copy_to_cpu()
            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            self._save_imgs(results, imgs_path[i:i + args.batch_size])
        logger.info("Finish")

    def _preprocess(self, img):
        return self.cfg.transforms(img)[0]

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def _save_imgs(self, results, imgs_path):
        for i in range(results.shape[0]):
            result = get_pseudo_color_map(results[i])
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))


def main(args):
    imgs_list, _ = get_image_list(args.image_path)

    # collect dynamic shape by auto_tune
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # create and run predictor
    predictor = Predictor(args)
    predictor.run(imgs_list)

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    if args.benchmark:
        predictor.autolog.report()


if __name__ == '__main__':
    args = parse_args()
    main(args)

~~~



运行结果：

~~~shell
2022-05-18 10:51:02 [INFO]	Use GPU
W0518 10:51:04.675190  3051 analysis_predictor.cc:795] The one-time configuration of analysis predictor failed, which may be due to native predictor called first and its configurations taken effect.
[libprotobuf ERROR /paddle/build/third_party/protobuf/src/extern_protobuf/src/google/protobuf/io/coded_stream.cc:208] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
--- Running analysis [ir_graph_build_pass]
[libprotobuf ERROR /paddle/build/third_party/protobuf/src/extern_protobuf/src/google/protobuf/io/coded_stream.cc:208] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
2022-05-18 10:51:07 [INFO]	(InvalidArgument) Failed to parse program_desc from binary string.
  [Hint: Expected desc_.ParseFromString(binary_str) == true, but received desc_.ParseFromString(binary_str):0 != true:1.] (at /paddle/paddle/fluid/framework/program_desc.cc:103)

2022-05-18 10:51:07 [INFO]	If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, ..., Expected all_dynamic_shape_set == true, ...', please set --enable_auto_tune=True to use auto_tune. 

~~~

[根据上面第6行及以下，搜索，可以得出模型加载出现问题的结论](https://aistudio.baidu.com/paddle/forum/topic/show/993858)，为了验证是否是连接中出现的问题，手动加载静态模型

```python
import paddle
paddle.jit.load('/home/aistudio/PaddleSeg/msmodels/model')
```

结果报错一致，由此断定为模型加载问题，搜索第三行[libprotobuf ERROR]([A protocol message was rejected because it was too big · Issue #166 · PaddlePaddle/Paddle (github.com)](https://github.com/PaddlePaddle/Paddle/issues/166))，可以找到16年Paddle遇到过类似的问题，是某个参数太大，大于67108864bytes=64MB，和周围同志交流过后，发现他们的模型都只有几M，推断为模型过大的问题，提issue得到回复，与推断一致。


  








