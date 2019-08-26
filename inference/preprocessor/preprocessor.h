#pragma once
#include <vector>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils/seg_conf_parser.h"

namespace  PaddleSolution {

class ImagePreProcessor {
protected:
    ImagePreProcessor() {};
    
public:
    virtual ~ImagePreProcessor() {}

    virtual bool single_process(const std::string& fname, float* data, int* ori_w, int* ori_h) = 0;

    virtual bool batch_process(const std::vector<std::string>& imgs, float* data, int* ori_w, int* ori_h) = 0;

}; // end of class ImagePreProcessor

std::shared_ptr<ImagePreProcessor> create_processor(const std::string &config_file);

} // end of namespace paddle_solution
