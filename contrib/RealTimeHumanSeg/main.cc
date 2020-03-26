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

#include <iostream>
#include <string>

#include "humanseg.h" // NOLINT
#include "humanseg_postprocess.h" // NOLINT

// Do predicting on a video file
int VideoPredict(const std::string& video_path, HumanSeg& seg)
{
  cv::VideoCapture capture;
  capture.open(video_path.c_str());
  if (!capture.isOpened()) {
    printf("can not open video : %s\n", video_path.c_str());
    return -1;
  }

  int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter video_out;
  std::string video_out_path = "result.avi";
  video_out.open(video_out_path.c_str(),
                 CV_FOURCC('M', 'J', 'P', 'G'),
                 30.0,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return -1;
  }

  cv::Mat frame;
  while (capture.read(frame)) {
    cv::Mat out_im = seg.Predict(frame);
    video_out.write(out_im);
  }
  capture.release();
  return 0;
}

// Do predicting on a image file
int ImagePredict(const std::string& image_path, HumanSeg& seg)
{
  cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
  cv::Mat out_im = seg.Predict(img);
  imwrite("result.jpeg", out_im);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cout << "Usage:"
              << "./humanseg ./models/ ./data/test.avi"
              << std::endl;
    return -1;
  }

  bool use_gpu = (argc == 4 ? std::stoi(argv[3]) : false);
  auto model_dir = std::string(argv[1]);
  auto input_path = std::string(argv[2]);

  // Init Model
  std::vector<float> means = {104.008, 116.669, 122.675};
  std::vector<float> scale = {1.000, 1.000, 1.000};
  HumanSeg seg(model_dir, means, scale, use_gpu);

  // Call ImagePredict while input_path is a image file path
  // The output will be saved as result.jpeg
  ImagePredict(input_path, seg);
  
  // Call VideoPredict while input_path is a video file path
  // The output will be saved as result.avi
  // VideoPredict(input_path, seg);
  return 0;
}
