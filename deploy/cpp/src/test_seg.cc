#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"
#include "yaml-cpp/yaml.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

DEFINE_string(model_dir, "", "Directory of the inference model. "
                             "It constains deploy.yaml and infer models");
DEFINE_string(img_path, "", "Path of the test image.");
DEFINE_string(devices, "GPU", "Use GPU or CPU devices. Default: GPU");
DEFINE_bool(use_trt, false, "Wether enable TensorRT when use GPU. Defualt: false.");
DEFINE_string(trt_precision, "fp32", "The precision of TensorRT, support fp32, fp16 and int8. Default: fp32");
DEFINE_bool(use_trt_dynamic_shape, false, "Wether enable dynamic shape when use GPU and TensorRT. Defualt: false.");
DEFINE_string(dynamic_shape_path, "", "If set dynamic_shape_path, it read the dynamic shape for TRT.");
DEFINE_bool(use_mkldnn, false, "Wether enable MKLDNN when use CPU. Defualt: false.");
DEFINE_string(save_dir, "", "Directory of the output image.");

typedef struct YamlConfig {
  std::string model_file;
  std::string params_file;
  bool is_normalize;
  bool is_resize;
  int resize_width;
  int resize_height;
}YamlConfig;

YamlConfig load_yaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);
  std::string model_file = node["Deploy"]["model"].as<std::string>();
  std::string params_file = node["Deploy"]["params"].as<std::string>();
  YamlConfig yaml_config = {model_file, params_file};
  if (node["Deploy"]["transforms"]) {
    const YAML::Node& transforms = node["Deploy"]["transforms"];
    for (size_t i = 0; i < transforms.size(); i++) {
      if (transforms[i]["type"].as<std::string>() == "Normalize") {
        yaml_config.is_normalize = true;
      } else if (transforms[i]["type"].as<std::string>() == "Resize") {
        yaml_config.is_resize = true;
        const YAML::Node& target_size = transforms[i]["target_size"];
        yaml_config.resize_width = target_size[0].as<int>();
        yaml_config.resize_height = target_size[1].as<int>();
      }
    }
  }
  return yaml_config;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const YamlConfig& yaml_config) {
  std::string& model_dir = FLAGS_model_dir;

  paddle_infer::Config infer_config;
  infer_config.SetModel(model_dir + "/" + yaml_config.model_file,
                  model_dir + "/" + yaml_config.params_file);
  infer_config.EnableMemoryOptim();

  if (FLAGS_devices == "CPU") {
    LOG(INFO) << "Use CPU";
    if (FLAGS_use_mkldnn) {
      LOG(INFO) << "Use MKLDNN";
      infer_config.EnableMKLDNN();
      infer_config.SetCpuMathLibraryNumThreads(5);
    }
  } else if(FLAGS_devices == "GPU") {
    LOG(INFO) << "Use GPU";
    infer_config.EnableUseGpu(100, 0);

    // TRT config
    if (FLAGS_use_trt) {
      LOG(INFO) << "Use TRT";
      LOG(INFO) << "trt_precision:" << FLAGS_trt_precision;

      // TRT precision
      if (FLAGS_trt_precision == "fp32") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kFloat32, false, false);
      } else if (FLAGS_trt_precision == "fp16") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kHalf, false, false);
      } else if (FLAGS_trt_precision == "int8") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kInt8, false, false);
      } else {
        LOG(FATAL) << "The trt_precision should be fp32, fp16 or int8.";
      }

      // TRT dynamic shape
      if (FLAGS_use_trt_dynamic_shape) {
        LOG(INFO) << "Enable TRT dynamic shape";
        if (FLAGS_dynamic_shape_path.empty()) {
          std::map<std::string, std::vector<int>> min_input_shape = {
              {"image", {1, 3, 112, 112}}};
          std::map<std::string, std::vector<int>> max_input_shape = {
              {"image", {1, 3, 1024, 2048}}};
          std::map<std::string, std::vector<int>> opt_input_shape = {
              {"image", {1, 3, 512, 1024}}};
          infer_config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                        opt_input_shape);
        } else {
          infer_config.EnableTunedTensorRtDynamicShape(FLAGS_dynamic_shape_path, true);
        }
      }
    }
  } else {
    LOG(FATAL) << "The devices should be GPU or CPU";
  }

  auto predictor = paddle_infer::CreatePredictor(infer_config);
  return predictor;
}

void hwc_img_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

cv::Mat read_process_image(const YamlConfig& yaml_config) {
  cv::Mat img = cv::imread(FLAGS_img_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  if (yaml_config.is_resize) {
    cv::resize(img, img, cv::Size(yaml_config.resize_width, yaml_config.resize_height));
  }
  if (yaml_config.is_normalize) {
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    img = (img - 0.5) / 0.5;
  }
  return img;
}


int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(FATAL) << "The model_dir should not be empty.";
  }

  // Load yaml
  std::string yaml_path = FLAGS_model_dir + "/deploy.yaml";
  YamlConfig yaml_config = load_yaml(yaml_path);

  // Prepare data
  cv::Mat img = read_process_image(yaml_config);
  int rows = img.rows;
  int cols = img.cols;
  int chs = img.channels();
  std::vector<float> input_data(1 * chs * rows * cols, 0.0f);
  hwc_img_2_chw_data(img, input_data.data());

  // Create predictor
  auto predictor = create_predictor(yaml_config);

  // Set input
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  std::vector<int> input_shape = {1, chs, rows, cols};
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input_data.data());

  // Run
  predictor->Run();

  // Get output
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  std::vector<int32_t> out_data(out_num);
  output_t->CopyToCpu(out_data.data());

  // Get pseudo image
  std::vector<uint8_t> out_data_u8(out_num);
  for (int i = 0; i < out_num; i++) {
    out_data_u8[i] = static_cast<uint8_t>(out_data[i]);
  }
  cv::Mat out_gray_img(output_shape[1], output_shape[2], CV_8UC1, out_data_u8.data());
  cv::Mat out_eq_img;
  cv::equalizeHist(out_gray_img, out_eq_img);
  cv::imwrite("out_img.jpg", out_eq_img);
  
  LOG(INFO) << "Finish, the result is saved in out_img.jpg";
}
