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

void SophgoInfer(const std::string& model_dir, const std::string& image_file) {
  std::string model_file = model_dir + "/pp_liteseg_1684x_f32.bmodel";
  std::string params_file;
  std::string config_file = model_dir + "/deploy.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseSophgo();
  auto model_format = fastdeploy::ModelFormat::SOPHGO;

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option, model_format);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  //model.GetPreprocessor().DisableNormalizeAndPermute();

  fastdeploy::TimeCounter tc;
  tc.Start();
  auto im_org = cv::imread(image_file);

  //the input of bmodel should be fixed
  int new_width = 512;
  int new_height = 512;
  cv::Mat im;
  cv::resize(im_org, im, cv::Size(new_width, new_height), cv::INTER_LINEAR);

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res);
  tc.End();
  tc.PrintInfo("PPSeg in Sophgo");

  cv::imwrite("infer_sophgo.jpg", vis_im);
  std::cout
      << "Visualized result saved in ./infer_sophgo.jpg"
      << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image run_option, "
           "e.g ./infer_model ./bmodel ./test.jpeg"
        << std::endl;
    return -1;
  }

  SophgoInfer(argv[1], argv[2]);
  return 0;
}

