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
#include <iostream>
#include <string>
#include "fastdeploy/vision.h"

void ONNXInfer(const std::string& model_dir, const std::string& image_file) {
  std::string model_file = model_dir + "/Portrait_PP_HumanSegV2_Lite_256x144_infer.onnx";
  std::string params_file;
  std::string config_file = model_dir + "/deploy.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto format = fastdeploy::ModelFormat::ONNX;

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option, format);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  fastdeploy::TimeCounter tc;
  tc.Start();
  auto im = cv::imread(image_file);
  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res);
  tc.End();
  tc.PrintInfo("PPSeg in ONNX");

  cv::imwrite("infer_onnx.jpg", vis_im);
  std::cout
      << "Visualized result saved in ./infer_onnx.jpg"
      << std::endl;
}

void RKNPU2Infer(const std::string& model_dir, const std::string& image_file) {
  std::string model_file = model_dir + "/Portrait_PP_HumanSegV2_Lite_256x144_infer_rk3588.rknn";
  std::string params_file;
  std::string config_file = model_dir + "/deploy.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseRKNPU2();
  auto format = fastdeploy::ModelFormat::RKNN;

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option, format);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  model.GetPreprocessor().DisablePermute();
  model.GetPreprocessor().DisableNormalize();

  fastdeploy::TimeCounter tc;
  tc.Start();
  auto im = cv::imread(image_file);
  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res);
  tc.End();
  tc.PrintInfo("PPSeg in RKNPU2");

  cv::imwrite("infer_rknn.jpg", vis_im);
  std::cout
      << "Visualized result saved in ./infer_rknn.jpg"
      << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./picodet_model_dir ./test.jpeg"
        << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    ONNXInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    RKNPU2Infer(argv[1], argv[2]);
  }
  return 0;
}

