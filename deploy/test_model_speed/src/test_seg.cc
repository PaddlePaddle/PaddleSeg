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

#include "common.h"

/*
Test the inference speed of segmentation models.
*/

DEFINE_string(model_dir, "", "Directory of inference models. ");
DEFINE_string(img_path, "", "Path of the test image.");
DEFINE_int32(target_width, 0, "The resized width of input image.");
DEFINE_int32(target_height, 0, "The resized height of input image.");
DEFINE_string(device, "GPU", "Use GPU or CPU device. Default: GPU");
DEFINE_bool(use_trt, false, "Wether enable TensorRT when use GPU. Defualt: false.");
DEFINE_string(trt_precision, "fp32", "The precision of TensorRT, support fp32, fp16 and int8. Default: fp32");
DEFINE_bool(use_trt_dynamic_shape, false, "Wether enable dynamic shape when use GPU and TensorRT. Defualt: false.");
DEFINE_string(dynamic_shape_path, "", "If set dynamic_shape_path, it read the dynamic shape for TRT.");
DEFINE_bool(use_mkldnn, false, "Wether enable MKLDNN when use CPU. Defualt: false.");
DEFINE_string(save_dir, "", "Directory of the output image.");

typedef struct Args {
  std::string model_dir;
  std::string img_path;
  int target_width;
  int target_height;

  std::string device;
  bool use_trt;
  std::string trt_precision;
  bool use_trt_dynamic_shape;
  std::string dynamic_shape_path;

  bool use_mkldnn;
  std::string save_dir;
}Args;

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
  YamlConfig yaml_cfg = {model_file, params_file};
  if (node["Deploy"]["transforms"]) {
    const YAML::Node& transforms = node["Deploy"]["transforms"];
    for (size_t i = 0; i < transforms.size(); i++) {
      if (transforms[i]["type"].as<std::string>() == "Normalize") {
        yaml_cfg.is_normalize = true;
      } else if (transforms[i]["type"].as<std::string>() == "Resize") {
        yaml_cfg.is_resize = true;
        const YAML::Node& target_size = transforms[i]["target_size"];
        yaml_cfg.resize_width = target_size[0].as<int>();
        yaml_cfg.resize_height = target_size[1].as<int>();
      }
    }
  }
  return yaml_cfg;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(const Args& args) {
  paddle_infer::Config infer_config;
  infer_config.SetModel(args.model_dir + "/model.pdmodel",
                  args.model_dir + "/model.pdiparams");
  infer_config.EnableMemoryOptim();

  if (args.device == "CPU") {
    LOG(INFO) << "Use CPU";
    if (args.use_mkldnn) {
      LOG(INFO) << "Use MKLDNN";
      infer_config.EnableMKLDNN();
      infer_config.SetCpuMathLibraryNumThreads(5);
    }
  } else if(args.device == "GPU") {
    LOG(INFO) << "Use GPU";
    infer_config.EnableUseGpu(100, 0);

    // TRT config
    if (args.use_trt) {
      LOG(INFO) << "Use TRT";
      LOG(INFO) << "trt_precision:" << args.trt_precision;

      // TRT precision
      if (args.trt_precision == "fp32") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kFloat32, false, false);
      } else if (args.trt_precision == "fp16") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kHalf, false, false);
      } else if (args.trt_precision == "int8") {
        infer_config.EnableTensorRtEngine(1 << 20, 1, 3,
          paddle_infer::PrecisionType::kInt8, false, false);
      } else {
        LOG(FATAL) << "The trt_precision should be fp32, fp16 or int8.";
      }

      // TRT dynamic shape
      if (args.use_trt_dynamic_shape) {
        LOG(INFO) << "Enable TRT dynamic shape";
        if (args.dynamic_shape_path.empty()) {
          std::map<std::string, std::vector<int>> min_input_shape = {
              {"image", {1, 3, 112, 112}}};
          std::map<std::string, std::vector<int>> max_input_shape = {
              {"image", {1, 3, 1024, 2048}}};
          std::map<std::string, std::vector<int>> opt_input_shape = {
              {"image", {1, 3, 512, 1024}}};
          infer_config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                        opt_input_shape);
        } else {
          infer_config.EnableTunedTensorRtDynamicShape(args.dynamic_shape_path, true);
        }
      }
    }
  } else {
    LOG(FATAL) << "The device should be GPU or CPU";
  }

  auto predictor = paddle_infer::CreatePredictor(infer_config);
  return predictor;
}

void hwc_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

cv::Mat read_image(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}

void pre_process_image(cv::Mat img, const Args& args, const YamlConfig& yaml_cfg, std::vector<float>& img_data, int& rows, int& cols, int& chs) {
  if (args.target_width != 0 && args.target_height != 0
      && (args.target_height != img.rows || args.target_width != img.cols)) {
    cv::resize(img, img, cv::Size(args.target_width, args.target_height));
  } else if (yaml_cfg.is_resize && yaml_cfg.resize_width != img.cols && yaml_cfg.resize_height != img.rows) {
    cv::resize(img, img, cv::Size(yaml_cfg.resize_width, yaml_cfg.resize_height));
  }

  if (yaml_cfg.is_normalize) {
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    img = (img - 0.5) / 0.5;
  }

  rows = img.rows;
  cols = img.cols;
  chs = img.channels();
  img_data.resize(rows * cols * chs);
  hwc_2_chw_data(img, img_data.data());
}

void run_infer(std::shared_ptr<paddle_infer::Predictor> predictor, const YamlConfig& yaml_cfg,
               const Args& args, bool save_res,
               Time* pre_time=nullptr, Time* run_time=nullptr) {
  // Prepare data
  cv::Mat img = read_image(args.img_path);
  int rows, cols, chs;
  std::vector<float> img_data;
  if (pre_time != nullptr) {
    pre_time->start();
  }
  pre_process_image(img, args, yaml_cfg, img_data, rows, cols, chs);
  if (pre_time != nullptr) {
    pre_time->stop();
  }
  //LOG(INFO) << "resized image: width is " << cols << " height is " << rows;

  if (run_time != nullptr) {
    run_time->start();
  }
  // Set input
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  std::vector<int> input_shape = {1, chs, rows, cols};
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(img_data.data());

  // Run
  predictor->Run();

  // Get output
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  std::vector<int64_t> out_data(out_num);
  output_t->CopyToCpu(out_data.data());
  if (run_time != nullptr) {
    run_time->stop();
  }

  // Get pseudo image
  if (save_res) {
    std::vector<uint8_t> out_data_u8(out_num);
    for (int i = 0; i < out_num; i++) {
      out_data_u8[i] = static_cast<uint8_t>(out_data[i]);
    }
    cv::Mat out_gray_img(output_shape[1], output_shape[2], CV_8UC1, out_data_u8.data());
    cv::Mat out_eq_img;
    cv::equalizeHist(out_gray_img, out_eq_img);
    cv::imwrite("out_img.jpg", out_eq_img);
    LOG(INFO) << "The result is saved in out_img.jpg";
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  Args args;
  args.model_dir = FLAGS_model_dir;
  args.img_path = FLAGS_img_path;
  args.target_width = FLAGS_target_width;
  args.target_height = FLAGS_target_height;
  args.device = FLAGS_device;
  args.use_trt = FLAGS_use_trt;
  args.trt_precision = FLAGS_trt_precision;
  args.use_trt_dynamic_shape = FLAGS_use_trt_dynamic_shape;
  args.dynamic_shape_path = FLAGS_dynamic_shape_path;
  args.use_mkldnn = FLAGS_use_mkldnn;
  args.save_dir = FLAGS_save_dir;

  if (args.model_dir == "") {
    LOG(FATAL) << "The model_dir should not be empty.";
  }

  // Load yaml
  YamlConfig yaml_cfg = load_yaml(args.model_dir + "/deploy.yaml");

  // Create predictor
  auto predictor = create_predictor(args);

  for (int i = 0; i < 50; i++) {
    run_infer(predictor, yaml_cfg, args, false);
  }

  Time pre_time, run_time;
  int repeat = 100;
  for (int i = 0; i < repeat; i++) {
    run_infer(predictor, yaml_cfg, args, false, &pre_time, &run_time);
  }
  LOG(INFO) << "Avg preprocess time: " << pre_time.used_time() / repeat << " ms";
  LOG(INFO) << "Avg run time: " << run_time.used_time() / repeat << " ms";
}
