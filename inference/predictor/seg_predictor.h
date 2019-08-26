#pragma once

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <paddle_inference_api.h>

#include <utils/seg_conf_parser.h>
#include <utils/utils.h>
#include <preprocessor/preprocessor.h>

namespace PaddleSolution {
    class Predictor {
        public:
            // init a predictor with a yaml config file
            int init(const std::string& conf);
            // predict api
            int predict(const std::vector<std::string>& imgs);
            
        private:
            int output_mask(
                const std::string& fname,
                float* p_out,
                int length,
                int* height = NULL,
                int* width = NULL);
            int native_predict(const std::vector<std::string>& imgs);
            int analysis_predict(const std::vector<std::string>& imgs);
        private:
            std::vector<float> _buffer;
            std::vector<int> _org_width;
            std::vector<int> _org_height;
            std::vector<std::string> _imgs_batch;
            std::vector<paddle::PaddleTensor> _outputs;

            std::vector<uchar> _mask;
            std::vector<uchar> _scoremap;

            PaddleSolution::PaddleSegModelConfigPaser _model_config;
            std::shared_ptr<PaddleSolution::ImagePreProcessor> _preprocessor;
            std::unique_ptr<paddle::PaddlePredictor> _main_predictor;
    };
}
