# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time

import numpy as np
from tqdm import tqdm
import paddle

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import paddleseg.transforms as T
from paddleseg.cvlibs import Config
from paddleseg.utils import logger, get_image_list, utils
from paddleseg.utils.visualize import get_pseudo_color_map
from export import SavedSegmentationNet
"""
Export the Paddle model to ONNX, infer the ONNX model by TRT.
Or, load the ONNX model and infer it by TRT.

Prepare:
* Install gpu driver, cuda toolkit and cudnn
* Install PaddlePaddle
* Install the requirements of PaddleSeg
* Download TensorRT 5/7 tar file according the version of cuda
* Install the trt whl in tar file, export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7/lib
* Run `pip install 'pycuda>=2019.1.1'`
* Run `pip install paddle2onnx onnx onnxruntime`

Usage:
    python deploy/python/infer_onnx_trt.py \
        --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml
    
    Please refer to following code for full usage.

Note:
* Some models are not supported exporting to ONNX.
* Some ONNX models are not supportd deploying by TRT.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--config", help="The config file.", type=str)
    parser.add_argument(
        "--model_path", help="The pretrained weights file.", type=str)
    parser.add_argument(
        "--onnx_model_path",
        help="If set onnx_model_path, it loads the onnx "
        "model and infer it by TRT",
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output/tmp')
    parser.add_argument(
        '--trt_version',
        help='The version of TRT that is 5 or 7',
        type=int,
        default=7)
    parser.add_argument('--width', help='width', type=int, default=1024)
    parser.add_argument('--height', help='height', type=int, default=512)
    parser.add_argument('--warmup', default=500, type=int, help='')
    parser.add_argument('--repeats', default=2000, type=int, help='')
    parser.add_argument(
        '--enable_profile', action='store_true', help='enable trt profile')
    parser.add_argument(
        '--print_model', action='store_true', help='print model to log')

    return parser.parse_args()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTPredictorV2(object):
    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    @staticmethod
    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    @staticmethod
    def trt7_do_inference(context,
                          bindings,
                          inputs,
                          outputs,
                          stream,
                          batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(
            batch_size=batch_size,
            bindings=bindings,
            stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs
        ]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    @staticmethod
    def trt7_do_inference_v2(args, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # warmup
        for _ in range(args.warmup):
            context.execute_async_v2(
                bindings=bindings, stream_handle=stream.handle)
        # Run inference.
        t_start = time.time()
        for _ in range(args.repeats):
            context.execute_async_v2(
                bindings=bindings, stream_handle=stream.handle)
        elapsed_time = time.time() - t_start
        latency = elapsed_time / args.repeats * 1000

        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs
        ]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs], latency

    @staticmethod
    def trt7_get_engine(onnx_file_path, input_shape, engine_file_path=""):
        TRT_LOGGER = trt.Logger()
        EXPLICIT_BATCH = 1 << (
            int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30
                builder.max_batch_size = 1
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print(
                        'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'
                        .format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(
                    onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None

                network.get_input(0).shape = input_shape
                #network.get_output(0).shape = [1, 19, 512, 1024]
                print('Completed parsing of ONNX file')
                print(
                    'Building an engine from file {}; this may take a while...'.
                    format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")

                if engine_file_path != "":
                    with open(engine_file_path, "wb") as f:
                        f.write(engine.serialize())
                    print("Save trt model in {}".format(engine_file_path))
                return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path,
                      "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    @staticmethod
    def trt7_run(args, onnx_file_path, input_data):
        engine_file_path = onnx_file_path[0:-5] + ".trt"
        input_shape = input_data.shape
        with TRTPredictorV2.trt7_get_engine(onnx_file_path, input_shape) as engine, \
            engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = TRTPredictorV2.allocate_buffers(
                engine)
            if args.enable_profile:
                context.profiler = trt.Profiler()

            # Do inference
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = input_data
            trt_outputs, latency = TRTPredictorV2.trt7_do_inference_v2(
                args,
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)
            return trt_outputs[0], latency

    @staticmethod
    def trt5_get_engine(onnx_file_path, engine_file_path=""):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        TRT_LOGGER = trt.Logger()

        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_workspace_size = 1 << 30  # 1GB
                builder.max_batch_size = 1
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found.'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(
                    onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    parser.parse(model.read())
                print('Completed parsing of ONNX file')

                print(
                    'Building an engine from file {}; this may take a while...'.
                    format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")

                if engine_file_path != "":
                    with open(engine_file_path, "wb") as f:
                        f.write(engine.serialize())
                        print("Save trt model in {}".format(engine_file_path))

                return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path,
                      "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    @staticmethod
    def trt5_do_inference(args,
                          context,
                          bindings,
                          inputs,
                          outputs,
                          stream,
                          batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # warmup
        for _ in range(args.warmup):
            context.execute_async(
                batch_size=batch_size,
                bindings=bindings,
                stream_handle=stream.handle)
        # Run inference.
        t_start = time.time()
        for _ in range(args.repeats):
            context.execute_async(
                batch_size=batch_size,
                bindings=bindings,
                stream_handle=stream.handle)
        elapsed_time = time.time() - t_start
        latency = elapsed_time / args.repeats * 1000

        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs
        ]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs], latency

    @staticmethod
    def trt5_run(args, onnx_file_path, input_data):
        engine_file_path = onnx_file_path[0:-5] + ".trt"
        input_shape = input_data.shape
        with TRTPredictorV2.trt5_get_engine(onnx_file_path) as engine, \
            engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = TRTPredictorV2.allocate_buffers(
                engine)
            if args.enable_profile:
                context.profiler = trt.Profiler()

            # Do inference
            # Set host input to the image. The common.do_inference function will
            # copy the input to the GPU before executing.
            inputs[0].host = input_data
            trt_outputs, latency = TRTPredictorV2.trt5_do_inference(
                args,
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream)
            return trt_outputs[0], latency


def run_paddle(paddle_model, input_data):
    paddle_model.eval()
    paddle_outs = paddle_model(paddle.to_tensor(input_data))
    out = paddle_outs[0].numpy()
    if out.ndim == 3:
        out = out[np.newaxis, :]
    return out


def check_and_run_onnx(onnx_model_path, input_data):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print('The onnx model has been checked.')

    ort_sess = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_sess.get_inputs()[0].name: input_data}
    ort_outs = ort_sess.run(None, ort_inputs)
    print("The onnx model has been predicted by ONNXRuntime.")

    return ort_outs[0]


def export_load_infer(args, model=None):
    """
    Export the ONNX model from PaddlePaddle, infer it by TRT.
    It checks the accuracy and tests the infer time.

    Args:
        args (dict): The input args.
        model (nn.Layer, optional): The paddle model to be exported and tested.
            If model is None, it creates a model with config file in args.
    """

    # 1. prepare
    if model is None:
        cfg = Config(args.config)
        model = cfg.model
    if args.model_path is not None:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    #model = SavedSegmentationNet(model)  # add argmax to the last layer
    model.eval()
    if args.print_model:
        print(model)

    input_shape = [1, 3, args.height, args.width]
    print("input shape:", input_shape)
    input_data = np.random.random(input_shape).astype('float32')
    model_name = os.path.basename(args.config).split(".")[0]

    # 2. run paddle
    paddle_out = run_paddle(model, input_data)
    print("out shape:", paddle_out.shape)
    print("The paddle model has been predicted by PaddlePaddle.\n")

    # 3. export onnx
    input_spec = paddle.static.InputSpec(input_shape, 'float32', 'x')
    onnx_model_path = os.path.join(args.save_dir, model_name + "_model")
    paddle.onnx.export(
        model, onnx_model_path, input_spec=[input_spec], opset_version=11)
    print("Completed export onnx model.\n")

    # 4. run and check onnx
    onnx_model_path = onnx_model_path + ".onnx"
    onnx_out = check_and_run_onnx(onnx_model_path, input_data)
    assert onnx_out.shape == paddle_out.shape
    np.testing.assert_allclose(onnx_out, paddle_out, rtol=0, atol=1e-03)
    print("The paddle and onnx models have the same outputs.\n")

    # 5. run and check trt
    assert args.trt_version in (5, 7), "trt_version should be 5 or 7"
    if args.trt_version == 5:
        trt_out, latency = TRTPredictorV2().trt5_run(args, onnx_model_path,
                                                     input_data)
    elif args.trt_version == 7:
        trt_out, latency = TRTPredictorV2().trt7_run(args, onnx_model_path,
                                                     input_data)
    print("trt avg latency: {:.3f} ms".format(latency))

    assert trt_out.size == paddle_out.size
    trt_out = trt_out.reshape(paddle_out.shape)
    np.testing.assert_allclose(trt_out, paddle_out, rtol=0, atol=1e-03)
    print("The paddle and trt models have the same outputs.\n")

    return latency


def load_infer(args):
    # Load the ONNX model and infer it by TRT

    input_shape = [1, 3, args.height, args.width]
    print("input shape:", input_shape)
    input_data = np.random.random(input_shape).astype('float32')

    # 1. check and run onnx
    onnx_model_path = args.onnx_model_path
    onnx_out = check_and_run_onnx(onnx_model_path, input_data)
    print("output shape:", onnx_out.shape, "\n")

    # 2. run and check trt
    assert args.trt_version in (5, 7), "trt_version should be 5 or 7"
    if args.trt_version == 5:
        trt_out, latency = TRTPredictorV2().trt5_run(args, onnx_model_path,
                                                     input_data)
    elif args.trt_version == 7:
        trt_out, latency = TRTPredictorV2().trt7_run(args, onnx_model_path,
                                                     input_data)
    print("trt avg latency: {:.3f} ms".format(latency))

    trt_out = trt_out.reshape(onnx_out.shape)
    np.testing.assert_allclose(trt_out, onnx_out, rtol=0, atol=1e-03)
    print("The onnx and trt models have the same outputs.\n")


if __name__ == '__main__':
    args = parse_args()
    if args.onnx_model_path is None:
        export_load_infer(args)
    else:
        load_infer(args)
