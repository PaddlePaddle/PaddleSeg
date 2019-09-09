#pragma once

#include <iostream>
#include <vector>
#include <string>

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
    }
}
