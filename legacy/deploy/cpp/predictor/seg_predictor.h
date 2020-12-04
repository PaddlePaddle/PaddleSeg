// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

#include <paddle_inference_api.h>
#include <opencv2/opencv.hpp>
#include "utils/seg_conf_parser.h"
#include "utils/utils.h"
#include "preprocessor/preprocessor.h"

namespace PaddleSolution {
class Predictor {
 public:
    // init a predictor with a yaml config file
    int init(const std::string& conf);
    // predict api
    int predict(const std::vector<std::string>& imgs);
 private:
    int output_mask(const std::string& fname, float* p_out, int length,
                int* height = NULL, int* width = NULL);
    int native_predict(const std::vector<std::string>& imgs);
    int analysis_predict(const std::vector<std::string>& imgs);
 private:
    std::vector<float> _buffer;
    std::vector<int> _org_width;
    std::vector<int> _org_height;
    std::vector<std::string> _imgs_batch;
    std::vector<paddle::PaddleTensor> _outputs;

    std::vector<uchar> _mask;
    std::vector<uchar> _scoremap;

    PaddleSolution::PaddleSegModelConfigPaser _model_config;
    std::shared_ptr<PaddleSolution::ImagePreProcessor> _preprocessor;
    std::unique_ptr<paddle::PaddlePredictor> _main_predictor;
    std::map<std::string, paddle::AnalysisConfig::Precision> TRT_MAP = {
        {"FP32", paddle::AnalysisConfig::Precision::kFloat32},
        {"FP16", paddle::AnalysisConfig::Precision::kHalf},
        {"INT8", paddle::AnalysisConfig::Precision::kInt8}
    };
};
}  // namespace PaddleSolution
