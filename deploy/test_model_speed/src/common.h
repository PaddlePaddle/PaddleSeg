#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

#include <stdarg.h>
#include <sys/stat.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#pragma once

class Time {
private:
  struct timeval _start_time;
  struct timeval _end_time;
  double _used_time;

public:
  Time() {
      _used_time = 0.0;
  }

  void start();
  void stop();
  void clear();
  double used_time();   // return the used time (ms)
};

cv::Mat read_image(const std::string& img_path);
void hwc_2_chw_data(const cv::Mat& hwc_img, float* data);

template<typename T>
std::string vector_2_str(std::vector<T> input) {
  std::stringstream ss;
  for (auto i : input) {
    ss << i << " ";
  }
  return ss.str();
}

bool file_exists(const std::string& path);