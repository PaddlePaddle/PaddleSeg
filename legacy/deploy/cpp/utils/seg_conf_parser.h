// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>

namespace PaddleSolution {
class PaddleSegModelConfigPaser {
 public:
    PaddleSegModelConfigPaser()
        :_class_num(0),
        _channels(0),
        _use_gpu(0),
        _use_pr(0),
        _batch_size(1),
        _model_file_name("__model__"),
        _param_file_name("__params__") {
    }
    ~PaddleSegModelConfigPaser() {
    }

    void reset() {
        _resize.clear();
        _mean.clear();
        _std.clear();
        _img_type.clear();
        _class_num = 0;
        _channels = 0;
        _use_gpu = 0;
        _use_pr = 0;
        _batch_size = 1;
        _model_file_name.clear();
        _model_path.clear();
        _param_file_name.clear();
        _trt_mode.clear();
    }

    std::string process_parenthesis(const std::string& str) {
        if (str.size() < 2) {
            return str;
        }
        std::string nstr(str);
        if (str[0] == '(' && str.back() == ')') {
            nstr[0] = '[';
            nstr[str.size() - 1] = ']';
        }
        return nstr;
    }

    template <typename T>
    std::vector<T> parse_str_to_vec(const std::string& str) {
        std::vector<T> data;
        auto node = YAML::Load(str);
        for (const auto& item : node) {
            data.push_back(item.as<T>());
        }
        return data;
    }

    bool load_config(const std::string& conf_file) {
        reset();
        YAML::Node config;
        try {
            config = YAML::LoadFile(conf_file);
        } catch(...) {
            return false;
        }
        // 1. get resize
        if (config["DEPLOY"]["EVAL_CROP_SIZE"].IsDefined()) {
            auto str = config["DEPLOY"]["EVAL_CROP_SIZE"].as<std::string>();
            _resize = parse_str_to_vec<int>(process_parenthesis(str));
        } else {
            std::cerr << "Please set EVAL_CROP_SIZE: (xx, xx)" << std::endl;
            return false;
        }

        // 2. get mean
        if (config["DEPLOY"]["MEAN"].IsDefined()) {
            for (const auto& item : config["DEPLOY"]["MEAN"]) {
                _mean.push_back(item.as<float>());
            }
        } else {
            std::cerr << "Please set MEAN: [xx, xx, xx]" << std::endl;
            return false;
        }

        // 3. get std
        if(config["DEPLOY"]["STD"].IsDefined()) {
            for (const auto& item : config["DEPLOY"]["STD"]) {
                _std.push_back(item.as<float>());
            }
        } else {
            std::cerr << "Please set STD: [xx, xx, xx]" << std::endl;
            return false;
        }

        // 4. get image type
		if (config["DEPLOY"]["IMAGE_TYPE"].IsDefined()) {
            _img_type = config["DEPLOY"]["IMAGE_TYPE"].as<std::string>();
        } else {
            std::cerr << "Please set IMAGE_TYPE: \"rgb\" or \"rgba\"" << std::endl;
            return false;
        }
        // 5. get class number
        if (config["DEPLOY"]["NUM_CLASSES"].IsDefined()) {
            _class_num = config["DEPLOY"]["NUM_CLASSES"].as<int>();
        } else {
            std::cerr << "Please set NUM_CLASSES: x" << std::endl;
            return false;
        }
        // 7. set model path
        if (config["DEPLOY"]["MODEL_PATH"].IsDefined()) {
            _model_path = config["DEPLOY"]["MODEL_PATH"].as<std::string>();
        } else {
            std::cerr << "Please set MODEL_PATH: \"/path/to/model_dir\"" << std::endl;
            return false;
        }
        // 8. get model file_name
        if (config["DEPLOY"]["MODEL_FILENAME"].IsDefined()) {
            _model_file_name = config["DEPLOY"]["MODEL_FILENAME"].as<std::string>();
        } else {
            _model_file_name = "__model__";
        }
        // 9. get model param file name
        if (config["DEPLOY"]["PARAMS_FILENAME"].IsDefined()) {
            _param_file_name
                = config["DEPLOY"]["PARAMS_FILENAME"].as<std::string>();
        } else {
            _param_file_name = "__params__";
        }
        // 10. get pre_processor
        if (config["DEPLOY"]["PRE_PROCESSOR"].IsDefined()) {
            _pre_processor = config["DEPLOY"]["PRE_PROCESSOR"].as<std::string>();
        } else {
            std::cerr << "Please set PRE_PROCESSOR: \"DetectionPreProcessor\"" << std::endl;
            return false;
        }
        // 11. use_gpu
        if (config["DEPLOY"]["USE_GPU"].IsDefined()) {
            _use_gpu = config["DEPLOY"]["USE_GPU"].as<int>();
        } else {
            _use_gpu = 0;
        }
        // 12. predictor_mode
        if (config["DEPLOY"]["PREDICTOR_MODE"].IsDefined()) {
            _predictor_mode = config["DEPLOY"]["PREDICTOR_MODE"].as<std::string>();
        } else {
            std::cerr << "Please set PREDICTOR_MODE: \"NATIVE\" or \"ANALYSIS\""  << std::endl;
            return false;
        }
        // 13. batch_size
        if (config["DEPLOY"]["BATCH_SIZE"].IsDefined()) {
            _batch_size = config["DEPLOY"]["BATCH_SIZE"].as<int>();
        } else {
            _batch_size = 1;
        }
        // 14. channels
        if (config["DEPLOY"]["CHANNELS"].IsDefined()) {
            _channels = config["DEPLOY"]["CHANNELS"].as<int>();
        } else {
            std::cerr << "Please set CHANNELS: x"  << std::endl;
            return false;
        }
        // 15. use_pr
        if (config["DEPLOY"]["USE_PR"].IsDefined()) {
            _use_pr = config["DEPLOY"]["USE_PR"].as<int>();
        } else {
            _use_pr = 0;
        }
        // 16. trt_mode
	if (config["DEPLOY"]["TRT_MODE"].IsDefined()) {
            _trt_mode = config["DEPLOY"]["TRT_MODE"].as<std::string>();
        } else {
            _trt_mode = "";
        }
        return true;
    }

    void debug() const {
        std::cout << "EVAL_CROP_SIZE: ("
                  << _resize[0] << ", " << _resize[1]
                  << ")" << std::endl;
        std::cout << "MEAN: [";
        for (int i = 0; i < _mean.size(); ++i) {
            if (i != _mean.size() - 1) {
                std::cout << _mean[i] << ", ";
            } else {
                std::cout << _mean[i];
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "STD: [";
        for (int i = 0; i < _std.size(); ++i) {
            if (i != _std.size() - 1) {
                std::cout << _std[i] << ", ";
            } else {
                std::cout << _std[i];
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "DEPLOY.IMAGE_TYPE: " << _img_type << std::endl;
        std::cout << "DEPLOY.NUM_CLASSES: " << _class_num << std::endl;
        std::cout << "DEPLOY.CHANNELS: " << _channels << std::endl;
        std::cout << "DEPLOY.MODEL_PATH: " << _model_path << std::endl;
        std::cout << "DEPLOY.MODEL_FILENAME: " << _model_file_name << std::endl;
        std::cout << "DEPLOY.PARAMS_FILENAME: "
                  << _param_file_name << std::endl;
        std::cout << "DEPLOY.PRE_PROCESSOR: " << _pre_processor << std::endl;
        std::cout << "DEPLOY.USE_GPU: " << _use_gpu << std::endl;
        std::cout << "DEPLOY.PREDICTOR_MODE: " << _predictor_mode << std::endl;
        std::cout << "DEPLOY.BATCH_SIZE: " << _batch_size << std::endl;
    }

    // DEPLOY.EVAL_CROP_SIZE
    std::vector<int> _resize;
    // DEPLOY.MEAN
    std::vector<float> _mean;
    // DEPLOY.STD
    std::vector<float> _std;
    // DEPLOY.IMAGE_TYPE
    std::string _img_type;
    // DEPLOY.NUM_CLASSES
    int _class_num;
    // DEPLOY.CHANNELS
    int _channels;
    // DEPLOY.MODEL_PATH
    std::string _model_path;
    // DEPLOY.MODEL_FILENAME
    std::string _model_file_name;
    // DEPLOY.PARAMS_FILENAME
    std::string _param_file_name;
    // DEPLOY.PRE_PROCESSOR
    std::string _pre_processor;
    // DEPLOY.USE_GPU
    int _use_gpu;
    // DEPLOY.PREDICTOR_MODE
    std::string _predictor_mode;
    // DEPLOY.BATCH_SIZE
    int _batch_size;
    // DEPLOY.USE_PR: OP Optimized model
    int _use_pr;
    // DEPLOY.TRT_MODE: TRT Precesion
    std::string _trt_mode;
};

}  // namespace PaddleSolution
