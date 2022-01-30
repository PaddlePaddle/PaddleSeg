#include "ort_session_handler.hpp"

namespace deploy {
OrtSessionHandler::OrtSessionHandler(const std::string &model_path,
                                     const std::vector<std::vector<int64_t>> &input_tensor_shapes)
    : _model_path(model_path),
      _input_tensor_shapes(input_tensor_shapes),
      _env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort session handler")),
      _session(nullptr) {
  Ort::SessionOptions session_options;

  // if onnxruntime is built with cuda provider, the following function can be added to use cuda gpu
  // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, gpu_index));

  std::basic_string<ORTCHAR_T> ort_model_path;
  std::copy(model_path.begin(), model_path.end(), std::back_inserter(ort_model_path));
  _session.reset(new Ort::Experimental::Session(_env, ort_model_path, session_options));

  if (_session->GetInputCount() != input_tensor_shapes.size()) {
    throw std::runtime_error("invalid input size");
  }
}

std::vector<float> OrtSessionHandler::preprocess(const cv::Mat &image, int target_height, int target_width,
                                                 const std::vector<float> &mean_val,
                                                 const std::vector<float> &std_val) const {
  if (image.empty() || image.channels() != 3) {
    throw std::runtime_error("invalid image");
  }

  if (target_height * target_width == 0) {
    throw std::runtime_error("invalid target dimension");
  }

  cv::Mat processed = image.clone();

  if (image.rows != target_height || image.cols != target_width) {
    cv::resize(processed, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
  }
  cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
  std::vector<float> data(3 * target_height * target_width);

  for (int i = 0; i < target_height; ++i) {
    for (int j = 0; j < target_width; ++j) {
      for (int c = 0; c < 3; ++c) {
        data[c * target_height * target_width + i * target_width + j] =
            (processed.data[i * target_width * 3 + j * 3 + c] / 255.0 - mean_val[c]) / std_val[c];
      }
    }
  }

  return data;
}
}  // namespace deploy
