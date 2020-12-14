// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "seg_predictor.h"
#undef min
namespace PaddleSolution {
    using std::chrono::duration_cast;
    int Predictor::init(const std::string& conf) {
        if (!_model_config.load_config(conf)) {
            LOG(FATAL) << "Fail to load config file: [" << conf << "]";
            return -1;
        }
        _preprocessor = PaddleSolution::create_processor(conf);
        if (_preprocessor == nullptr) {
            LOG(FATAL) << "Failed to create_processor";
            return -1;
        }

        int res_size = _model_config._resize[0] * _model_config._resize[1];
        _mask.resize(res_size);
        _scoremap.resize(res_size);

        bool use_gpu = _model_config._use_gpu;
        const auto& model_dir = _model_config._model_path;
        const auto& model_filename = _model_config._model_file_name;
        const auto& params_filename = _model_config._param_file_name;
        // load paddle model file
        if (_model_config._predictor_mode == "NATIVE") {
            paddle::NativeConfig config;
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.prog_file = prog_file;
            config.param_file = param_file;
            config.fraction_of_gpu_memory = 0;
            config.use_gpu = use_gpu;
            config.device = 0;
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else if (_model_config._predictor_mode == "ANALYSIS") {
            paddle::AnalysisConfig config;
            if (use_gpu) {
                config.EnableUseGpu(100, 0);
                if (TRT_MAP.find(_model_config._trt_mode) != TRT_MAP.end()) {
                    auto precision = TRT_MAP[_model_config._trt_mode];
                    bool use_cab = (precision == paddle::AnalysisConfig::Precision::kInt8);
                    config.EnableTensorRtEngine(1 << 30, _model_config._batch_size, 40,
                        precision, false, use_cab);
                }
            }
            auto prog_file = utils::path_join(model_dir, model_filename);
            auto param_file = utils::path_join(model_dir, params_filename);
            config.SetModel(prog_file, param_file);
            config.SwitchUseFeedFetchOps(false);
            config.SwitchSpecifyInputNames(true);
            config.EnableMemoryOptim();
            _main_predictor = paddle::CreatePaddlePredictor(config);
        } else {
            return -1;
        }
        return 0;
    }

    int Predictor::predict(const std::vector<std::string>& imgs) {
        if (_model_config._predictor_mode == "NATIVE") {
            return native_predict(imgs);
        } else if (_model_config._predictor_mode == "ANALYSIS") {
            return analysis_predict(imgs);
        }
        return -1;
    }

    int Predictor::output_mask(const std::string& fname, float* p_out,
                               int length, int* height, int* width) {
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        int eval_num_class = _model_config._class_num;

        int blob_out_len = length;
        int seg_out_len = eval_height * eval_width * eval_num_class;
        if (blob_out_len != seg_out_len) {
            LOG(ERROR) << " [FATAL] unequal: input vs output [" <<
                seg_out_len << "|" << blob_out_len << "]" << std::endl;
            return -1;
        }
        // post process
        _mask.clear();
        _scoremap.clear();
        std::vector<int> out_shape{eval_num_class, eval_height, eval_width};
        utils::argmax(p_out, out_shape, _mask, _scoremap);
        cv::Mat mask_png = cv::Mat(eval_height, eval_width, CV_8UC1);
        mask_png.data = _mask.data();
        std::string nname(fname);
        auto pos = fname.rfind(".");
        nname[pos] = '_';
        std::string mask_save_name = nname + "_mask.png";
        cv::imwrite(mask_save_name, mask_png);
        cv::Mat scoremap_png = cv::Mat(eval_height, eval_width, CV_8UC1);
        scoremap_png.data = _scoremap.data();
        std::string scoremap_save_name = nname + std::string("_scoremap.png");
        cv::imwrite(scoremap_save_name, scoremap_png);
        std::cout << "save mask of [" << fname << "] done" << std::endl;

        if (height && width) {
            int recover_height = *height;
            int recover_width = *width;
            cv::Mat recover_png = cv::Mat(recover_height, recover_width, CV_8UC1);
            cv::resize(scoremap_png, recover_png,
                cv::Size(recover_width, recover_height), 0, 0, cv::INTER_CUBIC);
            std::string recover_name = nname + std::string("_recover.png");
            cv::imwrite(recover_name, recover_png);
        }
        return 0;
    }

    int Predictor::native_predict(const std::vector<std::string>& imgs) {
        if (imgs.size() == 0) {
            LOG(ERROR) << "No image found";
            return -1;
        }
        int config_batch_size = _model_config._batch_size;

        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        std::size_t total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size,
                                          static_cast<int>(total_size));
        int batch = total_size / default_batch_size
                  + ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels
                              * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& org_width = _org_width;
        auto& org_height = _org_height;
        auto& imgs_batch = _imgs_batch;

        input_buffer.resize(batch_buffer_size);
        org_width.resize(default_batch_size);
        org_height.resize(default_batch_size);
        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels
                                 * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.resize(real_buffer_size);
            org_height.resize(batch_size);
            org_width.resize(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                org_width[i] = org_height[i] = 0;
            }
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }
            if (!_preprocessor->batch_process(imgs_batch,
                                              input_buffer.data(),
                                              org_width.data(),
                                              org_height.data())) {
                return -1;
            }
            paddle::PaddleTensor im_tensor;
            im_tensor.name = "image";
            if (!_model_config._use_pr) {
                im_tensor.shape = std::vector<int>{ batch_size, channels,
                                                eval_height, eval_width };
            } else {
                im_tensor.shape = std::vector<int>{ batch_size, eval_height,
                                                eval_width, channels};
            }
            im_tensor.data.Reset(input_buffer.data(),
                                 real_buffer_size * sizeof(float));
            im_tensor.dtype = paddle::PaddleDType::FLOAT32;
            feeds.push_back(im_tensor);
            _outputs.clear();
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!_main_predictor->Run(feeds, &_outputs, batch_size)) {
                LOG(ERROR) <<
                   "Failed: NativePredictor->Run() return false at batch: "
                 << u;
                continue;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>
                            (t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;
            int out_num = 1;
            // print shape of first output tensor for debugging
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < _outputs[0].shape.size(); ++j) {
                out_num *= _outputs[0].shape[j];
                std::cout << _outputs[0].shape[j] << ",";
            }
            std::cout << ")" << std::endl;

            size_t nums = _outputs.front().data.length() / sizeof(float);
            if (_model_config._use_pr) {
                nums = _outputs.front().data.length() / sizeof(int64_t);
            }
            // size mismatch checking
            bool size_mismatch = out_num % batch_size;
            size_mismatch |= (!_model_config._use_pr) && (nums != out_num);
            size_mismatch |= _model_config._use_pr && (nums != eval_height * eval_width);
            if (size_mismatch) {
                LOG(ERROR) << "output with a unexpected size";
                return -1;
            }

            if (_model_config._use_pr) {
                std::vector<uchar> out_data;
                out_data.resize(out_num);
                auto addr = reinterpret_cast<int64_t*>(_outputs[0].data.data());
                for (int r = 0; r < out_num; ++r) {
                    out_data[r] = (int)(addr[r]);
                }
                for (int r = 0; r < batch_size; ++r) {
                    cv::Mat mask_png = cv::Mat(eval_height, eval_width, CV_8UC1);
                    mask_png.data = out_data.data() + eval_height*eval_width*r;
                    auto name = imgs_batch[r];
                    auto pos = name.rfind(".");
                    name[pos] = '_';
                    std::string mask_save_name = name + "_mask.png";
                    cv::imwrite(mask_save_name, mask_png);
                }
                continue;
            }

            for (int i = 0; i < batch_size; ++i) {
                float* output_addr = reinterpret_cast<float*>(
                                    _outputs[0].data.data())
                                   + i * (nums / batch_size);
                output_mask(imgs_batch[i], output_addr,
                            nums / batch_size,
                            &org_height[i],
                            &org_width[i]);
            }
        }

        return 0;
    }

    int Predictor::analysis_predict(const std::vector<std::string>& imgs) {
        if (imgs.size() == 0) {
            LOG(ERROR) << "No image found";
            return -1;
        }

        int config_batch_size = _model_config._batch_size;
        int channels = _model_config._channels;
        int eval_width = _model_config._resize[0];
        int eval_height = _model_config._resize[1];
        auto total_size = imgs.size();
        int default_batch_size = std::min(config_batch_size,
                                          static_cast<int>(total_size));
        int batch = total_size / default_batch_size
                  + ((total_size % default_batch_size) != 0);
        int batch_buffer_size = default_batch_size * channels
                              * eval_width * eval_height;

        auto& input_buffer = _buffer;
        auto& org_width = _org_width;
        auto& org_height = _org_height;
        auto& imgs_batch = _imgs_batch;

        input_buffer.resize(batch_buffer_size);
        org_width.resize(default_batch_size);
        org_height.resize(default_batch_size);

        for (int u = 0; u < batch; ++u) {
            int batch_size = default_batch_size;
            if (u == (batch - 1) && (total_size % default_batch_size)) {
                batch_size = total_size % default_batch_size;
            }

            int real_buffer_size = batch_size * channels
                                 * eval_width * eval_height;
            std::vector<paddle::PaddleTensor> feeds;
            input_buffer.resize(real_buffer_size);
            org_height.resize(batch_size);
            org_width.resize(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                org_width[i] = org_height[i] = 0;
            }
            imgs_batch.clear();
            for (int i = 0; i < batch_size; ++i) {
                int idx = u * default_batch_size + i;
                imgs_batch.push_back(imgs[idx]);
            }

            if (!_preprocessor->batch_process(imgs_batch,
                                              input_buffer.data(),
                                              org_width.data(),
                                              org_height.data())) {
                return -1;
            }
            auto im_tensor = _main_predictor->GetInputTensor("image");
            if (!_model_config._use_pr) {
                im_tensor->Reshape({ batch_size, channels,
                                 eval_height, eval_width });
            } else {
                im_tensor->Reshape({ batch_size, eval_height,
                                eval_width, channels});
            }

            im_tensor->copy_from_cpu(input_buffer.data());

            auto t1 = std::chrono::high_resolution_clock::now();
            _main_predictor->ZeroCopyRun();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>
                            (t2 - t1).count();
            std::cout << "runtime = " << duration << std::endl;

            auto output_names = _main_predictor->GetOutputNames();
            auto output_t = _main_predictor->GetOutputTensor(
                                              output_names[0]);
            std::vector<int> output_shape = output_t->shape();

            int out_num = 1;
            std::cout << "size of outputs[" << 0 << "]: (";
            for (int j = 0; j < output_shape.size(); ++j) {
                out_num *= output_shape[j];
                std::cout << output_shape[j] << ",";
            }
            std::cout << ")" << std::endl;

            if (_model_config._use_pr) {
                    std::vector<int64_t> out_data;
                    out_data.resize(out_num);
                    output_t->copy_to_cpu(out_data.data());

                    std::vector<uchar> mask_data;
                    mask_data.resize(out_num);
                    auto addr = reinterpret_cast<int64_t*>(out_data.data());
                    for (int r = 0; r < out_num; ++r) {
                        mask_data[r] = (int)(addr[r]);
                    }
                    for (int r = 0; r < batch_size; ++r) {
                        cv::Mat mask_png = cv::Mat(eval_height, eval_width, CV_8UC1);
                        mask_png.data = mask_data.data() + eval_height*eval_width*r;
                        auto name = imgs_batch[r];
                        auto pos = name.rfind(".");
                        name[pos] = '_';
                        std::string mask_save_name = name + "_mask.png";
                        cv::imwrite(mask_save_name, mask_png);
                    }
                    continue;
            }

            std::vector<float> out_data;
            out_data.resize(out_num);
            output_t->copy_to_cpu(out_data.data());
            for (int i = 0; i < batch_size; ++i) {
                float* out_addr = out_data.data()
                               + (out_num / batch_size) * i;
                output_mask(imgs_batch[i], out_addr, out_num / batch_size,
                            &org_height[i], &org_width[i]);
            }
        }
        return 0;
    }
}  // namespace PaddleSolution
