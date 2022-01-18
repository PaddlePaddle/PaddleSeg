#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

// if onnxruntime is built with cuda provider, the following header can be added to use cuda gpu
// #include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

#include <numeric>
#include <opencv2/opencv.hpp>

namespace {
class OrtSessionHandler {
 public:
  /**
   *  @param model_path path to onnx model
   */
  OrtSessionHandler(const std::string &model_path, std::vector<std::vector<int64_t>> &input_tensor_shapes);

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

constexpr int BISENETV2_CITYSCAPES_IMAGE_HEIGHT = 1024;
constexpr int BISENETV2_CITYSCAPES_IMAGE_WIDTH = 1024;
constexpr int CITYSCAPES_NUM_CLASSES = 19;

static const std::vector<std::string> CITY_SCAPES_CLASSES = {
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky",  "person",   "rider",    "car",  "truck", "bus",  "train",         "motorcycle",   "bicycle"};

inline std::vector<cv::Scalar> to_cv_sccalar_colors(const std::vector<std::array<int, 3>> &colors) {
  std::vector<cv::Scalar> result;
  result.reserve(colors.size());
  std::transform(std::begin(colors), std::end(colors), std::back_inserter(result),
                 [](const auto &elem) { return cv::Scalar(elem[0], elem[1], elem[2]); });

  return result;
}

static const std::vector<cv::Scalar> CITYSCAPES_COLORS = to_cv_sccalar_colors({{128, 64, 128},
                                                                               {244, 35, 232},
                                                                               {70, 70, 70},
                                                                               {102, 102, 156},
                                                                               {190, 153, 153},
                                                                               {153, 153, 153},
                                                                               {250, 170, 30},
                                                                               {220, 220, 0},
                                                                               {107, 142, 35},
                                                                               {152, 251, 152},
                                                                               {70, 130, 180},
                                                                               {220, 20, 60},
                                                                               {255, 0, 0},
                                                                               {0, 0, 142},
                                                                               {0, 0, 70},
                                                                               {0, 60, 100},
                                                                               {0, 80, 100},
                                                                               {0, 0, 230},
                                                                               {119, 11, 32}});
}  // namespace

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: [app] [/path/to/image] [path/to/onnx/model]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string image_path = argv[1];
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cerr << "failed to load " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  const std::string onnx_model_path = argv[2];

  std::vector<std::vector<int64_t>> input_tensor_shapes{
      {1, 3, BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH}};
  OrtSessionHandler ort_session_handler(onnx_model_path, input_tensor_shapes);
  std::vector<float> input_data =
      ort_session_handler.preprocess(image, BISENETV2_CITYSCAPES_IMAGE_WIDTH, BISENETV2_CITYSCAPES_IMAGE_WIDTH);

  // output data's type might change for each different model
  auto output_data = ort_session_handler.run<int64_t>({input_data});

  // postprocess
  // this might change for each different model
  cv::Mat segm(BISENETV2_CITYSCAPES_IMAGE_HEIGHT, BISENETV2_CITYSCAPES_IMAGE_WIDTH, CV_8UC(3));
  for (int i = 0; i < BISENETV2_CITYSCAPES_IMAGE_HEIGHT; ++i) {
    cv::Vec3b *ptr_segm = segm.ptr<cv::Vec3b>(i);
    for (int j = 0; j < BISENETV2_CITYSCAPES_IMAGE_WIDTH; ++j) {
      const auto &color = CITYSCAPES_COLORS[output_data[0].first[i * BISENETV2_CITYSCAPES_IMAGE_WIDTH + j]];
      ptr_segm[j] = cv::Vec3b(color[0], color[1], color[2]);
    }
  }
  cv::resize(segm, segm, image.size(), 0, 0, cv::INTER_NEAREST);
  float blended_alpha = 0.4;
  segm = (1 - blended_alpha) * image + blended_alpha * segm;
  cv::imwrite("out_img.jpg", segm);

  return EXIT_SUCCESS;
}

namespace {
OrtSessionHandler::OrtSessionHandler(const std::string &model_path,
                                     std::vector<std::vector<int64_t>> &input_tensor_shapes)
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
}  // namespace
