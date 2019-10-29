#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef _WIN32
#include <filesystem>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

namespace PaddleSolution {
    namespace utils {
        inline std::string path_join(const std::string& dir, const std::string& path) {
            std::string seperator = "/";
            #ifdef _WIN32
            seperator = "\\";
            #endif
            return dir + seperator + path;
        }
        #ifndef _WIN32
        // scan a directory and get all files with input extensions
        inline std::vector<std::string> get_directory_images(const std::string& path, const std::string& exts)
        {
            std::vector<std::string> imgs;
            struct dirent *entry;
            DIR *dir = opendir(path.c_str());
            if (dir == NULL) {
                closedir(dir);
                return imgs;
            }

            while ((entry = readdir(dir)) != NULL) {
                std::string item = entry->d_name;
                auto ext = strrchr(entry->d_name, '.');
                if (!ext || std::string(ext) == "." || std::string(ext) == "..") {
                    continue;
                }
                if (exts.find(ext) != std::string::npos) {
                    imgs.push_back(path_join(path, entry->d_name));
                }
            }
            return imgs;
        }
        #else
        // scan a directory and get all files with input extensions
        inline std::vector<std::string> get_directory_images(const std::string& path, const std::string& exts)
        {
            std::vector<std::string> imgs;
            for (const auto& item : std::experimental::filesystem::directory_iterator(path)) {
                auto suffix = item.path().extension().string();
                if (exts.find(suffix) != std::string::npos && suffix.size() > 0) {
                    auto fullname = path_join(path, item.path().filename().string());
                    imgs.push_back(item.path().string());
                }
            }
            return imgs;
        }
        #endif

        // normalize and HWC_BGR -> CHW_RGB
        inline void normalize(cv::Mat& im, float* data, std::vector<float>& fmean, std::vector<float>& fstd) {
            int rh = im.rows;
            int rw = im.cols;
            int rc = im.channels();
            double normf = (double)1.0 / 255.0;
            #pragma omp parallel for
            for (int h = 0; h < rh; ++h) {
                const uchar* ptr = im.ptr<uchar>(h);
                int im_index = 0;
                for (int w = 0; w < rw; ++w) {
                    for (int c = 0; c < rc; ++c) {
                        int top_index = (c * rh + h) * rw + w;
                        float pixel = static_cast<float>(ptr[im_index++]);
                        pixel = (pixel * normf - fmean[c]) / fstd[c];
                        data[top_index] = pixel;
                    }
                }
            }
        }

        // argmax
        inline void argmax(float* out, std::vector<int>& shape, std::vector<uchar>& mask, std::vector<uchar>& scoremap) {
            int out_img_len = shape[1] * shape[2];
            int blob_out_len = out_img_len * shape[0];
            /*
            Eigen::TensorMap<Eigen::Tensor<float, 3>> out_3d(out, shape[0], shape[1], shape[2]);
            Eigen::Tensor<Eigen::DenseIndex, 2> argmax = out_3d.argmax(0);
            */
            float max_value = -1;
            int label = 0;
            #pragma omp parallel private(label)
            for (int i = 0; i < out_img_len; ++i) {
                max_value = -1;
                label = 0;
                #pragma omp for reduction(max : max_value)
                for (int j = 0; j < shape[0]; ++j) {
                    int index = i + j * out_img_len;
                    if (index >= blob_out_len) {
                        continue;
                    }
                    float value = out[index];
                    if (value > max_value) {
                        max_value = value;
                        label = j;
                    }
                }
                if (label == 0) max_value = 0;
                mask[i] = uchar(label);
                scoremap[i] = uchar(max_value * 255);
            }
        }
    }
}
