#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
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
Test the inference speed of detection models.
*/

DEFINE_string(model_dir, "", "Directory of different inference models. ");
DEFINE_string(img_path, "", "Path of the test image.");

DEFINE_string(device, "GPU", "Use GPU or CPU device. Default: GPU");
DEFINE_bool(use_trt, false, "Wether enable TensorRT when use GPU. Defualt: false.");
DEFINE_string(trt_precision, "fp32", "The precision of TensorRT, support fp32, fp16 and int8. Default: fp32");
DEFINE_bool(use_trt_dynamic_shape, true, "Wether enable dynamic shape when device=GPU, use_trt=True. Defualt: True.");
DEFINE_bool(use_trt_auto_tune, true, "Wether enable auto tune to collect dynamic shapes when device=GPU, use_trt=True, use_trt_dynamic_shape=True . Defualt: True.");
DEFINE_string(dynamic_shape_path, "./shape_range_info.pbtxt", "The tmp file to save the dynamic shape for auto tune.");

DEFINE_int32(warmup_iters, 50, "The iters for wamup.");
DEFINE_int32(run_iters, 100, "The iters for run.");

DEFINE_bool(use_mkldnn, false, "Wether enable MKLDNN when use CPU. Defualt: false.");
DEFINE_string(save_path, "./model_speed_det.txt", "Directory of the output image.");

typedef struct Args {
  std::string model_dir;
  std::string img_path;

  std::string device;
  bool use_trt;
  std::string trt_precision;
  bool use_trt_dynamic_shape;
  bool use_trt_auto_tune;
  std::string dynamic_shape_path;

  int warmup_iters;
  int run_iters;

  bool use_mkldnn;
  std::string save_path;
}Args;

typedef struct YamlConfig {
  bool use_dynamic_shape;

  bool is_resize;
  bool keep_ratio;
  int target_width;
  int target_height;

  bool is_normalize;
  std::vector<float> mean;
  std::vector<float> std;

  bool is_pad_stride;
  int stride;

  bool is_permute;

  void show() {
    LOG(INFO) << "use_dynamic_shape:" << use_dynamic_shape
              << ", is_resize:" << is_resize
              << ", keep_ratio:" << keep_ratio
              << ", target_width:" << target_width
              << ", target_height:" << target_height
              << ", is_normalize:" << is_normalize
              << ", is_pad_stride:" << is_pad_stride
              << ", stride:" << stride
              << ", is_permute:" << is_permute;
  }
}YamlConfig;

typedef struct DetInput {
  int batch_size;
  std::vector<float> img_data;
  std::vector<float> img_shape;
  std::vector<float> scale_factor;
  std::vector<float> in_net_shape;

  void show() {
    LOG(INFO) << "img_shape:" << vector_2_str<float>(img_shape)
              << ", scale_factor:" << vector_2_str<float>(scale_factor)
              << ", in_net_shape:" << vector_2_str<float>(in_net_shape);
  }
}DetInput;


YamlConfig load_yaml(const std::string& yaml_path) {
  YamlConfig yaml_cfg{false, 0, 0};
  YAML::Node node = YAML::LoadFile(yaml_path);

  if (node["use_dynamic_shape"].IsDefined()) {
    yaml_cfg.use_dynamic_shape = node["use_dynamic_shape"].as<bool>();
  } else {
    std::cerr << "Please set use_dynamic_shape in yaml." << std::endl;
  }

  if (node["Preprocess"].IsDefined()) {
    const YAML::Node& preprocess = node["Preprocess"];
    for (size_t i = 0; i < preprocess.size(); i++) {
      if (preprocess[i]["type"].as<std::string>() == "Resize") {
        yaml_cfg.is_resize = true;
        yaml_cfg.keep_ratio = preprocess[i]["keep_ratio"].as<bool>();
        const YAML::Node& target_size = preprocess[i]["target_size"];
        yaml_cfg.target_height = target_size[0].as<int>();
        yaml_cfg.target_width = target_size[1].as<int>();
      } else if (preprocess[i]["type"].as<std::string>() == "NormalizeImage") {
        yaml_cfg.is_normalize = true;
        yaml_cfg.mean = preprocess[i]["mean"].as<std::vector<float>>();
        yaml_cfg.std = preprocess[i]["std"].as<std::vector<float>>();
      } else if (preprocess[i]["type"].as<std::string>() == "PadStride") {
        yaml_cfg.is_pad_stride = true;
        yaml_cfg.stride = preprocess[i]["stride"].as<int>();
      } else if (preprocess[i]["type"].as<std::string>() == "Permute") {
        yaml_cfg.is_permute = true;
      } else {
        std::cerr << "Not support ops in preprocess:" 
          << preprocess[i]["type"].as<std::string>() << std::endl;
      }
    }
  } else {
    std::cerr << "Please set Preprocess in yaml." << std::endl;
  }
  
  return yaml_cfg;
}


void pre_process_image(cv::Mat img, const YamlConfig& yaml_cfg, DetInput& det_input) {
  // resize
  //cv::resize(img, img, cv::Size(yaml_cfg.target_width, yaml_cfg.target_height));
  if (yaml_cfg.is_resize) {
    std::pair<float, float> resize_scale; // w, h
    int origin_w = img.cols;
    int origin_h = img.rows;

    if (yaml_cfg.keep_ratio) {
      int im_size_max = std::max(origin_w, origin_h);
      int im_size_min = std::min(origin_w, origin_h);
      int target_size_max = std::max(yaml_cfg.target_width, yaml_cfg.target_height);
      int target_size_min = std::min(yaml_cfg.target_width, yaml_cfg.target_height);
      float scale_min =
          static_cast<float>(target_size_min) / static_cast<float>(im_size_min);
      float scale_max =
          static_cast<float>(target_size_max) / static_cast<float>(im_size_max);
      float scale_ratio = std::min(scale_min, scale_max);
      resize_scale = {scale_ratio, scale_ratio};
    } else {
      resize_scale.first =
          static_cast<float>(yaml_cfg.target_width) / static_cast<float>(origin_w);
      resize_scale.second =
          static_cast<float>(yaml_cfg.target_height) / static_cast<float>(origin_h);
    }

    cv::resize(img, img, cv::Size(), resize_scale.first, resize_scale.second);
    det_input.img_shape = {
        static_cast<float>(img.rows), static_cast<float>(img.cols),
    };
    det_input.in_net_shape = {
        static_cast<float>(img.rows), static_cast<float>(img.cols),
    };
    det_input.scale_factor = {resize_scale.second, resize_scale.first};
  }

  // normalize
  if (yaml_cfg.is_normalize) {
    std::vector<float> mean = yaml_cfg.mean;
    std::vector<float> std = yaml_cfg.std;
    img.convertTo(img, CV_32F, 1.0 / 255, 0);

    std::vector<cv::Mat> bgr_imgs(3);
    cv::split(img, bgr_imgs);
    for (auto i = 0; i < bgr_imgs.size(); i++) {
        bgr_imgs[i].convertTo(bgr_imgs[i], CV_32FC1, 1.0 / std[i], (0.0 - mean[i]) / std[i]);
    }
    cv::merge(bgr_imgs, img);
  }
  
  // pad stride
  if (yaml_cfg.is_pad_stride) {
    int rh = img.rows;
    int rw = img.cols;
    int stride = yaml_cfg.stride;
    int nh = (rh / stride) * stride + (rh % stride != 0) * stride;
    int nw = (rw / stride) * stride + (rw % stride != 0) * stride;
    cv::copyMakeBorder(
        img, img, 0, nh - rh, 0, nw - rw, cv::BORDER_CONSTANT, cv::Scalar(0));
    det_input.in_net_shape = {
        static_cast<float>(img.rows), static_cast<float>(img.cols),
    };
  }

  // permute
  if (yaml_cfg.is_permute) {
    det_input.img_data.resize(img.rows * img.cols * img.channels());
    hwc_2_chw_data(img, det_input.img_data.data());
  }
}


void auto_tune(const Args& args, const YamlConfig& yaml_cfg) {
  paddle_infer::Config infer_config;
  infer_config.SetModel(args.model_dir + "/model.pdmodel",
                  args.model_dir + "/model.pdiparams");
  infer_config.EnableUseGpu(100, 0);
  infer_config.EnableTensorRtEngine(1 << 10, 1, 3,
          paddle_infer::PrecisionType::kFloat32, false, false);
  infer_config.CollectShapeRangeInfo(args.dynamic_shape_path);
  infer_config.DisableGlogInfo();
  auto predictor = paddle_infer::CreatePredictor(infer_config);

  // Prepare data
  cv::Mat ori_img = read_image(args.img_path);
  cv::Mat img = ori_img.clone();

  DetInput det_input;
  det_input.batch_size = 1;
  pre_process_image(img, yaml_cfg, det_input);
  det_input.show();

  // Set input
  auto input_names = predictor->GetInputNames();
  auto input_shape = predictor->GetInputHandle(input_names[0]);
  input_shape->Reshape(std::vector<int>{det_input.batch_size, 2});
  input_shape->CopyFromCpu(det_input.img_shape.data());

  auto input_img = predictor->GetInputHandle(input_names[1]);
  input_img->Reshape(std::vector<int>{det_input.batch_size, 3, det_input.in_net_shape[0], det_input.in_net_shape[1]});
  input_img->CopyFromCpu(det_input.img_data.data());

  auto input_scale = predictor->GetInputHandle(input_names[2]);
  input_scale->Reshape(std::vector<int>{det_input.batch_size, 2});
  input_scale->CopyFromCpu(det_input.scale_factor.data());

  // Run
  predictor->Run();
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(const Args& args, const YamlConfig& yaml_cfg) {
  paddle_infer::Config infer_config;
  infer_config.SetModel(args.model_dir + "/model.pdmodel",
                  args.model_dir + "/model.pdiparams");
  infer_config.EnableMemoryOptim();
  infer_config.DisableGlogInfo();

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
      LOG(INFO) << "Trt_precision: " << args.trt_precision;

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
      bool use_dynamic_shape = args.use_trt_dynamic_shape || yaml_cfg.use_dynamic_shape;
      if (use_dynamic_shape) {
        LOG(INFO) << "Enable TRT dynamic shape";
        if (args.dynamic_shape_path.empty()) {
          LOG(INFO) << "Use manual dynamic shape";
          std::map<std::string, std::vector<int>> min_input_shape = {
              {"image", {1, 3, 112, 112}}};
          std::map<std::string, std::vector<int>> max_input_shape = {
              {"image", {1, 3, 1024, 2048}}};
          std::map<std::string, std::vector<int>> opt_input_shape = {
              {"image", {1, 3, 512, 1024}}};
          infer_config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                        opt_input_shape);
        } else {
          LOG(INFO) << "Load auto tune dynamic shape";
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

void run_infer(std::shared_ptr<paddle_infer::Predictor> predictor,
               const Args& args, const YamlConfig& yaml_cfg,
               Time* pre_time=nullptr, Time* run_time=nullptr) {
  // Prepare data
  cv::Mat ori_img = read_image(args.img_path);
  cv::Mat img = ori_img.clone();

  DetInput det_input;
  det_input.batch_size = 1;

  if (pre_time != nullptr) {
    pre_time->start();
  }
  pre_process_image(img, yaml_cfg, det_input);
  if (pre_time != nullptr) {
    pre_time->stop();
  }
  //LOG(INFO) << "resized image: width is " << cols << " height is " << rows;

  if (run_time != nullptr) {
    run_time->start();
  }
  // Set input
  auto input_names = predictor->GetInputNames();
  auto input_shape = predictor->GetInputHandle(input_names[0]);
  input_shape->Reshape(std::vector<int>{det_input.batch_size, 2});
  input_shape->CopyFromCpu(det_input.img_shape.data());

  auto input_img = predictor->GetInputHandle(input_names[1]);
  input_img->Reshape(std::vector<int>{det_input.batch_size, 3, det_input.in_net_shape[0], det_input.in_net_shape[1]});
  input_img->CopyFromCpu(det_input.img_data.data());

  auto input_scale = predictor->GetInputHandle(input_names[2]);
  input_scale->Reshape(std::vector<int>{det_input.batch_size, 2});
  input_scale->CopyFromCpu(det_input.scale_factor.data());

  // Run
  predictor->Run();

  // Get output
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  std::vector<float> out_data(out_num);
  output_t->CopyToCpu(out_data.data());
  if (run_time != nullptr) {
    run_time->stop();
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  Args args;
  args.model_dir = FLAGS_model_dir;
  args.img_path = FLAGS_img_path;
  args.device = FLAGS_device;
  args.use_trt = FLAGS_use_trt;
  args.trt_precision = FLAGS_trt_precision;
  args.use_trt_dynamic_shape = FLAGS_use_trt_dynamic_shape;
  args.use_trt_auto_tune = FLAGS_use_trt_auto_tune;
  args.dynamic_shape_path = FLAGS_dynamic_shape_path;
  args.warmup_iters = FLAGS_warmup_iters;
  args.run_iters = FLAGS_run_iters;
  args.use_mkldnn = FLAGS_use_mkldnn;
  args.save_path = FLAGS_save_path;

  // Load yaml
  YamlConfig yaml_cfg = load_yaml(args.model_dir + "/infer_cfg.yml");
  yaml_cfg.show();

  if (args.device == "GPU" && args.use_trt && (args.use_trt_dynamic_shape || yaml_cfg.use_dynamic_shape) && args.use_trt_auto_tune) {
    LOG(INFO) << "-----Auto tune-----";
    auto_tune(args, yaml_cfg);
  }

  LOG(INFO) << "-----Create predictor-----";
  auto predictor = create_predictor(args, yaml_cfg);

  LOG(INFO) << "-----Warmup-----";
  for (int i = 0; i < args.warmup_iters; i++) {
    run_infer(predictor, args, yaml_cfg);
  }

  LOG(INFO) << "-----Run-----";
  Time pre_time, run_time;
  for (int i = 0; i < args.run_iters; i++) {
    run_infer(predictor, args, yaml_cfg, &pre_time, &run_time);
  }

  LOG(INFO) << "Avg preprocess time: " << pre_time.used_time() / args.run_iters << " ms";
  LOG(INFO) << "Avg run time: " << run_time.used_time() / args.run_iters << " ms";

  std::size_t found = args.model_dir.find_last_of("/\\");
  std::string model_name = args.model_dir.substr(found + 1);
  std::ofstream ofs(args.save_path, std::ios::out | std::ios::app);
  ofs << "| " << model_name << " | " << pre_time.used_time() / args.run_iters
      << " | " << run_time.used_time() / args.run_iters
      << " | " << std::endl;
  ofs.close();
}
