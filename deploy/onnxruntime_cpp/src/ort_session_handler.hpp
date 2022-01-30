#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

// if onnxruntime is built with cuda provider, the following header can be added to use cuda gpu
// #include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

#include <opencv2/opencv.hpp>

namespace deploy {
class OrtSessionHandler {
 public:
  /**
   *  @param model_path path to onnx model
   */
  OrtSessionHandler(const std::string &model_path, const std::vector<std::vector<int64_t>> &input_tensor_shapes);

  virtual std::vector<float> preprocess(const cv::Mat &image, int target_height, int target_width,
                                        const std::vector<float> &mean_val = {0.5, 0.5, 0.5},
                                        const std::vector<float> &std_val = {0.5, 0.5, 0.5}) const;

  /**
   *   @file function to get output tensors
   *   @brief each std::pair<DataType *, std::vector<int64_t>> is a pair of output tensor's data and its dimension
   *   most semantic segmentation networks will have only one output tensor
   */
  template <typename DataType = float>
  std::vector<std::pair<DataType *, std::vector<int64_t>>> run(const std::vector<std::vector<float>> &input_data) const;

 private:
  std::string _model_path;
  std::vector<std::vector<int64_t>> _input_tensor_shapes;
  Ort::Env _env;
  std::unique_ptr<Ort::Experimental::Session> _session;
};

template <typename DataType>
std::vector<std::pair<DataType *, std::vector<int64_t>>> OrtSessionHandler::run(
    const std::vector<std::vector<float>> &input_data) const {
  if (_session->GetInputCount() != input_data.size()) {
    throw std::runtime_error("invalid input size");
  }

  std::vector<Ort::Value> input_tensors;
  for (int i = 0; i < _session->GetInputCount(); ++i) {
    input_tensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(
        const_cast<float *>(input_data[i].data()), input_data[i].size(), _input_tensor_shapes[i]));
  }

  std::vector<Ort::Value> output_tensors =
      _session->Run(_session->GetInputNames(), input_tensors, _session->GetOutputNames());

  std::vector<std::pair<DataType *, std::vector<int64_t>>> output(_session->GetOutputCount());
  std::vector<std::vector<int64_t>> output_shapes = _session->GetOutputShapes();
  for (int i = 0; i < _session->GetOutputCount(); ++i) {
    output[i] = std::make_pair(std::move(output_tensors[i].GetTensorMutableData<DataType>()), output_shapes[i]);
  }

  return output;
}
}  // namespace deploy
