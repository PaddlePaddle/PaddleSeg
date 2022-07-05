// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "paddle_inference_api.h"  // NOLINT

#include "core/general-server/general_model_service.pb.h"
#include "core/general-server/op/general_infer_helper.h"
#include "core/predictor/tools/pp_shitu_tools/preprocess_op.h"


namespace baidu {
namespace paddle_serving {
namespace serving {

class GeneralSegOp
    : public baidu::paddle_serving::predictor::OpWithChannel<GeneralBlob> {
 public:
  typedef std::vector<paddle::PaddleTensor> TensorVector;

  DECLARE_OP(GeneralSegOp);

  int inference();

 private:
    // clas preprocess
    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> std_ = {0.5f, 0.5f, 0.5f};
    float scale_ = 0.00392157;

    Feature::Normalize normalize_op_;
    Feature::Permute permute_op_;

    // read pics
    cv::Mat Base2Mat(std::string &base64_data);
    std::string base64Decode(const char* Data, int DataByte);

};

}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
