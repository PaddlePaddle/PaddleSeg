# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from paddle.fluid.contrib.slim.quantization.quantization_pass import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization.quantization_pass import AddQuantDequantPass
from paddle.fluid.contrib.slim.quantization.quantization_pass import _op_real_in_out_name
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
import paddle.fluid as fluid
import os

import HumanSeg.utils.logging as logging


class HumanSegPostTrainingQuantization(PostTrainingQuantization):
    def __init__(self,
                 executor,
                 dataset,
                 program,
                 inputs,
                 outputs,
                 batch_size=10,
                 batch_nums=None,
                 scope=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        '''
        The class utilizes post training quantization methon to quantize the
        fp32 model. It uses calibrate data to calculate the scale factor of
        quantized variables, and inserts fake quant/dequant op to obtain the
        quantized model.

        Args:
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            dataset(Python Iterator): The data Reader.
            program(fluid.Program): The paddle program, save the parameters for model.
            inputs(dict): The input of prigram.
            outputs(dict): The output of program.
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of
                calibrate data is batch_size*batch_nums. If batch_nums is None, use
                all data provided by sample_generator as calibrate data.
            scope(fluid.Scope, optional): The scope of the program, use it to load
                and save variables. If scope=None, get scope by global_scope().
            algo(str, optional): If algo=KL, use KL-divergenc method to
                get the more precise scale factor. If algo='direct', use
                abs_max methon to get the scale factor. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops
                that will be quantized. Default is ["conv2d", "depthwise_conv2d",
                "mul"].
            is_full_quantized(bool, optional): If set is_full_quantized as True,
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type
                according to the input quantizable_op_type.
            is_use_cache_file(bool, optional): If set is_use_cache_file as False,
                all temp data will be saved in memory. If set is_use_cache_file as True,
                it will save temp data to disk. When the fp32 model is complex or
                the number of calibrate data is large, we should set is_use_cache_file
                as True. Defalut is False.
            cache_dir(str, optional): When is_use_cache_file is True, set cache_dir as
                the directory for saving temp data. Default is ./temp_post_training.
        Returns:
            None
        '''
        self._executor = executor
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._scope = fluid.global_scope() if scope == None else scope
        self._algo = algo
        self._is_use_cache_file = is_use_cache_file
        self._cache_dir = cache_dir
        if self._is_use_cache_file and not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        supported_quantizable_op_type = \
            QuantizationTransformPass._supported_quantizable_op_type + \
            AddQuantDequantPass._supported_quantizable_op_type
        if is_full_quantize:
            self._quantizable_op_type = supported_quantizable_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in supported_quantizable_op_type + \
                    AddQuantDequantPass._activation_type, \
                    op_type + " is not supported for quantization."

        self._place = self._executor.place
        self._program = program
        self._feed_list = list(inputs.values())
        self._fetch_list = list(outputs.values())
        self._data_loader = None

        self._op_real_in_out_name = _op_real_in_out_name
        self._bit_length = 8
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._sampling_data = {}
        self._quantized_var_scale_factor = {}

    def quantize(self):
        '''
        Quantize the fp32 model. Use calibrate data to calculate the scale factor of
        quantized variables, and inserts fake quant/dequant op to obtain the
        quantized model.

        Args:
            None
        Returns:
            the program of quantized model.
        '''
        self._preprocess()

        batch_id = 0
        for data in self._data_loader():
            self._executor.run(
                program=self._program,
                feed=data,
                fetch_list=self._fetch_list,
                return_numpy=False)
            self._sample_data(batch_id)

            if batch_id % 5 == 0:
                logging.info("run batch: {}".format(batch_id))
            batch_id += 1
            if self._batch_nums and batch_id >= self._batch_nums:
                break
        logging.info("all run batch: ".format(batch_id))
        logging.info("calculate scale factor ...")
        self._calculate_scale_factor()
        logging.info("update the program ...")
        self._update_program()

        self._save_output_scale()
        return self._program

    def save_quantized_model(self, save_model_path):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model
        Returns:
            None
        '''
        feed_vars_names = [var.name for var in self._feed_list]
        fluid.io.save_inference_model(
            dirname=save_model_path,
            feeded_var_names=feed_vars_names,
            target_vars=self._fetch_list,
            executor=self._executor,
            params_filename='__params__',
            main_program=self._program)

    def _preprocess(self):
        '''
        Load model and set data loader, collect the variable names for sampling,
        and set activation variables to be persistable.
        '''
        feed_vars = [fluid.framework._get_var(var.name, self._program) \
            for var in self._feed_list]

        self._data_loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        self._data_loader.set_sample_list_generator(
            self._dataset.generator(self._batch_size, drop_last=True),
            places=self._place)

        # collect the variable names for sampling
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        for op in self._program.global_block().ops:
            op_type = op.type
            if op_type in self._quantizable_op_type:
                if op_type in ("conv2d", "depthwise_conv2d"):
                    self._quantized_act_var_name.add(op.input("Input")[0])
                    self._quantized_weight_var_name.add(op.input("Filter")[0])
                    self._quantized_act_var_name.add(op.output("Output")[0])
                elif op_type == "mul":
                    if self._is_input_all_not_persistable(
                            op, persistable_var_names):
                        op._set_attr("skip_quant", True)
                        logging.warning(
                            "Skip quant a mul op for two input variables are not persistable"
                        )
                    else:
                        self._quantized_act_var_name.add(op.input("X")[0])
                        self._quantized_weight_var_name.add(op.input("Y")[0])
                        self._quantized_act_var_name.add(op.output("Out")[0])
                else:
                    # process other quantizable op type, the input must all not persistable
                    if self._is_input_all_not_persistable(
                            op, persistable_var_names):
                        input_output_name_list = self._op_real_in_out_name[
                            op_type]
                        for input_name in input_output_name_list[0]:
                            for var_name in op.input(input_name):
                                self._quantized_act_var_name.add(var_name)
                        for output_name in input_output_name_list[1]:
                            for var_name in op.output(output_name):
                                self._quantized_act_var_name.add(var_name)

        # set activation variables to be persistable, so can obtain
        # the tensor data in sample_data
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = True
