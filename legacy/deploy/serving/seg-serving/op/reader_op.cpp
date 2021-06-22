// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>

#include "predictor/framework/memory.h"
#include "seg-serving/op/reader_op.h"
#include "seg-serving/op/seg_conf.h"
namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::image_segmentation::Request;
using baidu::paddle_serving::image_segmentation::ImageSegReqItem;
using baidu::utils::seg_conf::SegConf;
int ReaderOp::inference() {
  const Request* req = dynamic_cast<const Request*>(get_request_message());
//  LOG(INFO) << "Receive request in dense service:" << req->ShortDebugString();

  ReaderOutput* res = mutable_data<ReaderOutput>();
  if (!res) {
    LOG(ERROR) << "Failed get op tls reader object output";
    return -1;
  }

  TensorVector* in = &res->tensors;
  uint32_t batch_size = req->instances_size();
  if (batch_size <= 0) {
    LOG(WARNING) << "No instances need to inference!";
    return -1;
  }

  static const SegConf *sc_ptr = SegConf::instance();
  std::vector<double> pmean;
  if(sc_ptr->get_mean_vector(pmean) != 0) {
   LOG(ERROR) << "Can't load the mean items";
   return -1;
  }

  std::vector<double> scale;
  if(sc_ptr->get_std_vector(scale) != 0) {
    LOG(ERROR) << "Can't load the scale items";
    return -1;
  }

  std::vector<int> iresize;
  if(sc_ptr->get_size_vector(iresize) != 0) {
    LOG(ERROR) << "Can't load size vector";
    return -1;
  }

  int channels;
  if(sc_ptr->get_channels(channels) != 0) {
    LOG(ERROR) << "Can't load channels";
    return -1;
  }

  //bool enable_crop = SegConf._enable_crop;

  cv::Size resize;
  resize.height = iresize[1];
  resize.width = iresize[0];

  paddle::PaddleTensor in_tensor;
  in_tensor.name = "image";
  in_tensor.dtype = paddle::FLOAT32;
  // shape assignment
  in_tensor.shape.push_back(batch_size);  // batch_size
  in_tensor.shape.push_back(channels);
  in_tensor.shape.push_back(resize.height);
  in_tensor.shape.push_back(resize.width);

  // tls resource assignment
  size_t dense_capacity = channels * resize.width * resize.height;
  size_t len = dense_capacity * sizeof(float) * batch_size;

  // Allocate buffer in PaddleTensor, so that buffer will be managed by the
  // Tensor
  in_tensor.data.Resize(len);
  float* data = reinterpret_cast<float*>(in_tensor.data.data());
  if (in_tensor.data.data() == NULL) {
    LOG(ERROR) << "Failed create temp float array, "
               << "size=" << dense_capacity * batch_size * sizeof(float);
    return -1;
  }
  std::vector<int> *in_width_vec = &res->width_vec;
  std::vector<int> *in_height_vec = &res->height_vec;

  for (uint32_t si = 0; si < batch_size; si++) {
    // parse image object from x-image
    const ImageSegReqItem& ins = req->instances(si);
    // read dense image from request bytes
    const char* binary = ins.image_binary().c_str();
    //size_t length = ins.image_length();
    size_t length = ins.image_binary().length();
    if (length == 0) {
      LOG(ERROR) << "Empty image, length is 0";
      return -1;
    }

    _image_vec_tmp.clear();
    _image_vec_tmp.assign(binary, binary + length);
    _image_8u_tmp = cv::imdecode(cv::Mat(_image_vec_tmp),
                CV_LOAD_IMAGE_UNCHANGED);
    if (_image_8u_tmp.data == NULL) {
      LOG(ERROR) << "Image decode failed!";
      return -1;
    }

    // accumulate length
    const int HH = _image_8u_tmp.rows;
    const int WW = _image_8u_tmp.cols;
    const int CC = _image_8u_tmp.channels();

   //HH: cols WW:rows
    in_width_vec->push_back(HH);
    in_height_vec->push_back(WW);

    // resize
    if (_image_8u_tmp.cols != resize.width ||
        _image_8u_tmp.rows != resize.height) {
          cv::Mat resize_image;
          cv::resize(_image_8u_tmp, resize_image, resize);
          _image_8u_tmp = resize_image;
          LOG(INFO) << "Succ crop one image[CHW=" << _image_8u_tmp.channels()
                    << ", " << _image_8u_tmp.cols << ", " << _image_8u_tmp.rows
                    << "]"
                    << " from image[CHW=" << CC << ", " << HH << ", " << WW << "]";
    }

    // BGR->RGB transformer
    //cv::cvtColor(_image_8u_tmp, _image_8u_rgb, cv::COLOR_GRAY2BGR);
    _image_8u_rgb = _image_8u_tmp;

    const int H = _image_8u_rgb.rows;
    const int W = _image_8u_rgb.cols;
    const int C = _image_8u_rgb.channels();
    if (H != resize.height || W != resize.width || C != channels) {
      LOG(ERROR) << "Image " << si << " has incompitable size (" << H << ", " << W << "," << C << ")";
      return -1;
    }

    LOG(INFO) << "Succ read one image, C: " << C << ", W: " << W
              << ", H: " << H;

    float* data_ptr = data + dense_capacity * si;
    for (int h = 0; h < H; h++) {
      // p points to a new line
      unsigned char* p = _image_8u_rgb.ptr<unsigned char>(h);
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          // HWC(row,column,channel) -> CWH
          data_ptr[W * H * c + W * h + w] =
              (p[C * w + c] - pmean[c]) / scale[c];
         //HWC->CHW
         //data_ptr[W * H * c + w * H + h] =
         //     (p[C * w + c] - pmean[c]) / scale[c];

        }
      }
    }
  }
  in->push_back(in_tensor);

  return 0;
}

DEFINE_OP(ReaderOp);

}  // namespace serving
}  // namespace paddle_serving
}  // namespace baidu
