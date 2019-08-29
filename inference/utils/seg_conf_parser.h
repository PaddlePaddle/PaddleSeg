#pragma once
#include <iostream>
#include <vector>
#include <string>

#include <yaml-cpp/yaml.h>
namespace PaddleSolution {

    class PaddleSegModelConfigPaser {
    public:
        PaddleSegModelConfigPaser()
            :_class_num(0),
            _channels(0),
            _use_gpu(0),
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
            _batch_size = 1;
            _model_file_name.clear();
            _model_path.clear();
            _param_file_name.clear();
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

            YAML::Node config = YAML::LoadFile(conf_file);
            // 1. get resize
            auto str = config["DEPLOY"]["EVAL_CROP_SIZE"].as<std::string>();
            _resize = parse_str_to_vec<int>(process_parenthesis(str));

            // 2. get mean
            for (const auto& item : config["DEPLOY"]["MEAN"]) {
                _mean.push_back(item.as<float>());
            }

            // 3. get std
            for (const auto& item : config["DEPLOY"]["STD"]) {
                _std.push_back(item.as<float>());
            }

            // 4. get image type
            _img_type = config["DEPLOY"]["IMAGE_TYPE"].as<std::string>();
            // 5. get class number
            _class_num = config["DEPLOY"]["NUM_CLASSES"].as<int>();
            // 7. set model path
            _model_path = config["DEPLOY"]["MODEL_PATH"].as<std::string>();
            // 8. get model file_name
            _model_file_name = config["DEPLOY"]["MODEL_FILENAME"].as<std::string>();
            // 9. get model param file name
            _param_file_name = config["DEPLOY"]["PARAMS_FILENAME"].as<std::string>();
            // 10. get pre_processor
            _pre_processor = config["DEPLOY"]["PRE_PROCESSOR"].as<std::string>();
            // 11. use_gpu
            _use_gpu = config["DEPLOY"]["USE_GPU"].as<int>();
            // 12. predictor_mode
            _predictor_mode = config["DEPLOY"]["PREDICTOR_MODE"].as<std::string>();
            // 13. batch_size
            _batch_size = config["DEPLOY"]["BATCH_SIZE"].as<int>();
            // 14. channels
            _channels = config["DEPLOY"]["CHANNELS"].as<int>();
            return true;
        }

        void debug() const {

            std::cout << "EVAL_CROP_SIZE: (" << _resize[0] << ", " << _resize[1] << ")" << std::endl;

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
                }
                else {
                    std::cout << _std[i];
                }
            }
            std::cout << "]" << std::endl;

            std::cout << "DEPLOY.IMAGE_TYPE: " << _img_type << std::endl;
            std::cout << "DEPLOY.NUM_CLASSES: " << _class_num << std::endl;
            std::cout << "DEPLOY.CHANNELS: " << _channels << std::endl;
            std::cout << "DEPLOY.MODEL_PATH: " << _model_path << std::endl;
            std::cout << "DEPLOY.MODEL_FILENAME: " << _model_file_name << std::endl;
            std::cout << "DEPLOY.PARAMS_FILENAME: " << _param_file_name << std::endl;
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
    };

}
