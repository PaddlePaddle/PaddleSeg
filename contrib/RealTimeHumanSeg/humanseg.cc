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

# include "humanseg.h"
# include "humanseg_postprocess.h"

// Normalize the image by (pix - mean) * scale
void NormalizeImage(
    const std::vector<float> &mean,
    const std::vector<float> &scale,
    cv::Mat& im, // NOLINT
    float* input_buffer) {
  int height = im.rows;
  int width = im.cols;
  int stride = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base = h * width + w;
      input_buffer[base + 0 * stride] =
          (im.at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      input_buffer[base + 1 * stride] =
          (im.at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      input_buffer[base + 2 * stride] =
          (im.at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

// Load Model and return model predictor
void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    std::unique_ptr<paddle::PaddlePredictor>* predictor) {
  // Config the model info
  paddle::AnalysisConfig config;
  config.SetModel(model_dir);
  if (use_gpu) {
      config.EnableUseGpu(100, 0);
  } else {
      config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // Memory optimization
  config.EnableMemoryOptim();
  *predictor = std::move(CreatePaddlePredictor(config));
}

void HumanSeg::Preprocess(const cv::Mat& image_mat) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(192, 192), 0.f, 0.f, cv::INTER_LINEAR);

  im.convertTo(im, CV_32FC3, 1.0);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  input_shape_ = {1, rc, rh, rw};
  input_data_.resize(1 * rc * rh * rw);
  float* buffer = input_data_.data();
  NormalizeImage(mean_, scale_, im, input_data_.data());
}

cv::Mat HumanSeg::Postprocess(const cv::Mat& im) {
  int h = input_shape_[2];
  int w = input_shape_[3];
  scoremap_data_.resize(3 * h * w * sizeof(float));
  float* base = output_data_.data() + h * w;
  for (int i = 0; i < h * w; ++i) {
    scoremap_data_[i] = uchar(base[i] * 255);
  }

  cv::Mat im_scoremap = cv::Mat(h, w, CV_8UC1);
  im_scoremap.data = scoremap_data_.data();
  cv::resize(im_scoremap, im_scoremap, cv::Size(im.cols, im.rows));
  im_scoremap.convertTo(im_scoremap, CV_32FC1, 1 / 255.0);

  float* pblob = reinterpret_cast<float*>(im_scoremap.data);
  int out_buff_capacity = 10 * im.cols * im.rows * sizeof(float);
  segout_data_.resize(out_buff_capacity);
  unsigned char* seg_result = segout_data_.data();
  MergeProcess(im.data, pblob, im.rows, im.cols, seg_result);
  cv::Mat seg_mat(im.rows, im.cols, CV_8UC1, seg_result);
  cv::resize(seg_mat, seg_mat, cv::Size(im.cols, im.rows));
  cv::GaussianBlur(seg_mat, seg_mat, cv::Size(5, 5), 0, 0);
  float fg_threshold = 0.8;
  float bg_threshold = 0.4;
  cv::Mat show_seg_mat;
  seg_mat.convertTo(seg_mat, CV_32FC1, 1 / 255.0);
  ThresholdMask(seg_mat, fg_threshold, bg_threshold, show_seg_mat);
  auto out_im = MergeSegMat(show_seg_mat, im);
  return out_im;
}

cv::Mat HumanSeg::Predict(const cv::Mat& im) {
  // Preprocess image
  Preprocess(im);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // Run predictor
  predictor_->ZeroCopyRun();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[0]);
  auto output_shape = out_tensor->shape();
  // Calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  // Postprocessing result
  return Postprocess(im);
}
