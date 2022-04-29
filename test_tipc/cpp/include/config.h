// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "include/utility.h"


class SegConfig {
public:
  explicit SegConfig(const std::string &config_file) {
    config_map_ = LoadConfig(config_file);

    this->use_gpu = bool(stoi(config_map_["use_gpu"]));

    this->gpu_id = stoi(config_map_["gpu_id"]);

    this->gpu_mem = stoi(config_map_["gpu_mem"]);

    this->cpu_math_library_num_threads =
        stoi(config_map_["cpu_math_library_num_threads"]);

    this->use_mkldnn = bool(stoi(config_map_["use_mkldnn"]));

    this->use_tensorrt = bool(stoi(config_map_["use_tensorrt"]));
    this->use_fp16 = bool(stoi(config_map_["use_fp16"]));

    this->model_path.assign(config_map_["model_path"]);
    this->params_path.assign(config_map_["params_path"]);

    this->is_resize = bool(stoi(config_map_["is_resize"]));
    this->is_normalize = bool(stoi(config_map_["is_normalize"]));
    
    if (this->is_resize) {
      this->resize_width = stoi(config_map_["resize_width"]);
      this->resize_height = stoi(config_map_["resize_height"]);
    }
    
  }

  bool use_gpu = false;

  int gpu_id = 0;

  int gpu_mem = 4000;

  int cpu_math_library_num_threads = 1;

  bool use_mkldnn = false;

  bool use_tensorrt = false;
  bool use_fp16 = false;

  std::string model_path;

  std::string params_path;

  bool is_resize = false;
  bool is_normalize = true;

  int resize_width = 2048;
  int resize_height = 1024;

  void PrintConfigInfo();

private:
  // Load configuration
  std::map<std::string, std::string> LoadConfig(const std::string &config_file);

  std::vector<std::string> split(const std::string &str,
                                 const std::string &delim);

  std::map<std::string, std::string> config_map_;
};

