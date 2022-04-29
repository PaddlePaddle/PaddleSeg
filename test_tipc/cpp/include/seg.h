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

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/preprocess_op.h>

using namespace paddle_infer;

class Segmentor {
public:
  explicit Segmentor(const std::string &model_path,
                      const std::string &params_path, const bool &use_gpu,
                      const int &gpu_id, const int &gpu_mem,
                      const int &cpu_math_library_num_threads,
                      const bool &use_mkldnn, const bool &use_tensorrt,
                      const bool &use_fp16, const bool &is_normalize,
                      const bool &is_resize,
                      const int &resize_width, const int &resize_height) {
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    this->use_mkldnn_ = use_mkldnn;
    this->use_tensorrt_ = use_tensorrt;
    this->use_fp16_ = use_fp16;

    this->is_resize_ = is_resize;
    this->is_normalize_ = is_normalize;
    this->resize_width_ = resize_width;
    this->resize_height_ = resize_height;

    LoadModel(model_path, params_path);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_path, const std::string &params_path);

  // Run predictor
  std::tuple<std::vector<int64>, std::vector<int>, double> Run(cv::Mat &img);

private:
  std::shared_ptr<Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;
  bool use_tensorrt_ = false;
  bool use_fp16_ = false;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;

  bool is_resize_ = false;
  bool is_normalize_ = true;
  int resize_width_ = 2048;
  int resize_height_ = 1024;

  // pre-process
  ResizeImg resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;
};

