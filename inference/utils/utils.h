#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace PaddleSolution {
    namespace utils {
        inline std::string path_join(const std::string& dir, const std::string& path) {
            std::string seperator = "/";
            #ifdef _WIN32
            seperator = "\\";
            #endif
            return dir + seperator + path;
        }

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
    }
}
