//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

#include "paddle_inference_api.h" // NOLINT

// Load Paddle Inference Model
void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    std::unique_ptr<paddle::PaddlePredictor>* predictor);

class HumanSeg {
 public:
  explicit HumanSeg(const std::string& model_dir,
                        const std::vector<float>& mean,
                        const std::vector<float>& scale,
                        bool use_gpu = false) :
      mean_(mean),
      scale_(scale) {
    LoadModel(model_dir, use_gpu, &predictor_);
  }

  // Run predictor
  cv::Mat Predict(const cv::Mat& im);

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& im);
  // Postprocess result
  cv::Mat Postprocess(const cv::Mat& im);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<int> input_shape_;
  std::vector<float> output_data_;
  std::vector<uchar> scoremap_data_;
  std::vector<uchar> segout_data_;
  std::vector<float> mean_;
  std::vector<float> scale_;
};
