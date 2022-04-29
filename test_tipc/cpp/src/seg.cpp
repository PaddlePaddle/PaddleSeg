// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <include/seg.h>


void Segmentor::LoadModel(const std::string &model_path,
                           const std::string &params_path) {
  paddle_infer::Config config;
  config.SetModel(model_path, params_path);

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
    if (this->use_tensorrt_) {
      config.EnableTensorRtEngine(
          1 << 20, 1, 3,
          this->use_fp16_ ? paddle_infer::Config::Precision::kHalf
                          : paddle_infer::Config::Precision::kFloat32,
          false, false);
    }
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  config.SwitchUseFeedFetchOps(false);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePredictor(config);
}


std::tuple<std::vector<int64>, std::vector<int>, double> Segmentor::Run(cv::Mat &img) {
  cv::Mat new_img;
  img.copyTo(new_img);

  // Transforms
  if (this->is_resize_) {
    this->resize_op_.Run(img, new_img, this->resize_width_, this->resize_height_);
  }
  if (this->is_normalize_) {
    this->normalize_op_.Run(&new_img, this->mean_, this->scale_,
                          this->is_scale_);
  }
  
  std::vector<float> input(1 * 3 * new_img.rows * new_img.cols, 0.0f);
  this->permute_op_.Run(&new_img, input.data());
  
  // Set input
  auto input_names = this->predictor_->GetInputNames();
  auto input_t = this->predictor_->GetInputHandle(input_names[0]);
  input_t->Reshape({1, 3, new_img.rows, new_img.cols});
  auto start = std::chrono::system_clock::now();
  input_t->CopyFromCpu(input.data());

  // Run
  this->predictor_->Run();

  // Get output
  std::vector<int64> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
  std::vector<int> out_shape = output_t->shape();
  int out_num = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                std::multiplies<int>());
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double cost_time = double(duration.count()) *
                     std::chrono::microseconds::period::num /
                     std::chrono::microseconds::period::den;

  std::tuple<std::vector<int64>, std::vector<int>, double> result = std::make_tuple(out_data, out_shape, cost_time);
  return result;
}

