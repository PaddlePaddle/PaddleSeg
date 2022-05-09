#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "common.h"

void Time::start() {
    gettimeofday(&_start_time, NULL);
}

void Time::stop() {
    gettimeofday(&_end_time, NULL);
    _used_time += (_end_time.tv_sec - _start_time.tv_sec) * 1000.0 + (double)(_end_time.tv_usec - _start_time.tv_usec) / 1000.0;
}

void Time::clear() {
    _used_time = 0.0;
}

double Time::used_time() {
    return _used_time;
}

cv::Mat read_image(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}

void hwc_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

bool file_exists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}