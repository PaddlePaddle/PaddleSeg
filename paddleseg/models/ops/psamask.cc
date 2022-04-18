#include "paddle/extension.h"

#include <vector>

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif


template <typename data_t>
void psamask_collect_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  for(int i{0}; i<num_*feature_H_*feature_H_*feature_W_*feature_W_; ++i) {
      out_data[i] = 0;
  }
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
        for(int w = 0; w < feature_W_; w++) {
            const int hstart = max(0, half_mask_H_ - h);
            const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
            const int wstart = max(0, half_mask_W_ - w);
            const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
            for (int hidx = hstart; hidx < hend; hidx++) {
                for (int widx = wstart; widx < wend; widx++) {
                    out_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w] = x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
                }
            }
        }
    }
  }
}


template <typename data_t>
void psamask_distribute_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  for(int i{0}; i<num_*feature_H_*feature_H_*feature_W_*feature_W_; ++i) {
      out_data[i] = 0;
  }
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
        for(int w = 0; w < feature_W_; w++) {
            const int hstart = max(0, half_mask_H_ - h);
            const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
            const int wstart = max(0, half_mask_W_ - w);
            const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
            for (int hidx = hstart; hidx < hend; hidx++) {
                for (int widx = wstart; widx < wend; widx++) {
                    out_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)] = x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
                }
            }
        }
    }
  }
}


template <typename data_t>
void psamask_collect_backward_kernel(const data_t* grad_out_data,
                            data_t* grad_x_data,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  for (int i{0}; i < num_ * mask_H_ * mask_W_ * feature_H_ * feature_W_; ++i){
      grad_x_data[i] = 0;
  }
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
        for(int w = 0; w < feature_W_; w++) {
            const int hstart = max(0, half_mask_H_ - h);
            const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
            const int wstart = max(0, half_mask_W_ - w);
            const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
            for (int hidx = hstart; hidx < hend; hidx++) {
                for (int widx = wstart; widx < wend; widx++) {
                    grad_x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] = grad_out_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w];
                }
            }
        }
    }
  }
}


template <typename data_t>
void psamask_distribute_backward_kernel(const data_t* grad_out_data,
                            data_t* grad_x_data,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  for (int i{0}; i < num_ * mask_H_ * mask_W_ * feature_H_ * feature_W_; ++i){
      grad_x_data[i] = 0;
  }
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
        for(int w = 0; w < feature_W_; w++) {
            const int hstart = max(0, half_mask_H_ - h);
            const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
            const int wstart = max(0, half_mask_W_ - w);
            const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
            for (int hidx = hstart; hidx < hend; hidx++) {
                for (int widx = wstart; widx < wend; widx++) {
                    grad_x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] = grad_out_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)];
                }
            }
        }
    }
  }
}


std::vector<paddle::Tensor> PSAMaskCPUForward(const paddle::Tensor& x,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  CHECK_INPUT(x);

  auto out = paddle::Tensor(paddle::PlaceType::kCPU, std::vector<int64_t>{num_, feature_H_ * feature_W_, feature_H_, feature_W_});

  if (psa_type == 0) {
    PD_DISPATCH_FLOATING_TYPES(
        x.type(), "psamask_collect_forward_kernel", ([&] {
            psamask_collect_forward_kernel<data_t>(
                x.data<data_t>(), out.mutable_data<data_t>(x.place()), num_, feature_H_, feature_W_,
                            mask_H_, mask_W_, half_mask_H_, half_mask_W_);
        }));
  }
  else{
      PD_DISPATCH_FLOATING_TYPES(
        x.type(), "psamask_distribute_forward_kernel", ([&] {
            psamask_distribute_forward_kernel<data_t>(
                x.data<data_t>(),  out.mutable_data<data_t>(x.place()), num_, feature_H_, feature_W_,
                            mask_H_, mask_W_, half_mask_H_, half_mask_W_);
        }));
  }

  return {out};
}


std::vector<paddle::Tensor> PSAMaskCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  CHECK_INPUT(x);
  CHECK_INPUT(out);
  CHECK_INPUT(grad_out);

  auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());
  
  if (psa_type == 0) {
    PD_DISPATCH_FLOATING_TYPES(out.type(), "psamask_collect_backward_kernel", ([&] {
                                psamask_collect_backward_kernel<data_t>(
                                    grad_out.data<data_t>(),
                                    grad_x.mutable_data<data_t>(x.place()),
                                    num_, feature_H_, feature_W_,
                             mask_H_, mask_W_, half_mask_H_, half_mask_W_);
                                }));
  } else {
      PD_DISPATCH_FLOATING_TYPES(out.type(), "psamask_distribute_backward_kernel", ([&] {
                                psamask_distribute_backward_kernel<data_t>(
                                    grad_out.data<data_t>(),
                                    grad_x.mutable_data<data_t>(x.place()),
                                    num_, feature_H_, feature_W_,
                             mask_H_, mask_W_, half_mask_H_, half_mask_W_);
                                }));
  }

  return {grad_x};
}


#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> PSAMaskCUDAForward(const paddle::Tensor& x,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_);
std::vector<paddle::Tensor> PSAMaskCUDABackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_);
#endif


std::vector<paddle::Tensor> PSAMaskForward(const paddle::Tensor& x,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  if (x.place() == paddle::PlaceType::kCPU) {
    return PSAMaskCPUForward(x, psa_type, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_);
#ifdef PADDLE_WITH_CUDA
  } else if (x.place() == paddle::PlaceType::kGPU) {
    return PSAMaskCUDAForward(x, psa_type, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_);
#endif
  } else {
    PD_THROW("Unsupported device type for forward function of custom relu operator.");
  }
}


std::vector<paddle::Tensor> PSAMaskBackward(const paddle::Tensor& x,
                                         const paddle::Tensor& out,
                                         const paddle::Tensor& grad_out,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  if (x.place() == paddle::PlaceType::kCPU) {
    return PSAMaskCPUBackward(x, out, grad_out, psa_type, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_);
#ifdef PADDLE_WITH_CUDA
  } else if (x.place() == paddle::PlaceType::kGPU) {
    return PSAMaskCUDABackward(x, out, grad_out, psa_type, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_);
#endif
  } else {
    PD_THROW("Unsupported device type for backward function of custom relu operator.");
  }
}


std::vector<std::vector<int64_t>> PSAMaskInferShape(const std::vector<int64_t> x_shape) {
    return {std::vector<int64_t>{x_shape[0], x_shape[2] * x_shape[3], x_shape[2], x_shape[3]}};
}


std::vector<paddle::DataType> PSAMaskInferDtype(paddle::DataType x_dtype) {
  return {x_dtype};
}


PD_BUILD_OP(psamask)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({
        "psa_type: int",
        "num_: int",
        "feature_H_: int",
        "feature_W_: int",
        "mask_H_: int",
        "mask_W_: int",
        "half_mask_H_: int",
        "half_mask_W_: int"})
    .SetKernelFn(PD_KERNEL(PSAMaskForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PSAMaskInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PSAMaskInferDtype));

PD_BUILD_GRAD_OP(psamask)
    .Inputs({"X", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .Attrs({
        "psa_type: int",
        "num_: int",
        "feature_H_: int",
        "feature_W_: int",
        "mask_H_: int",
        "mask_W_: int",
        "half_mask_H_: int",
        "half_mask_W_: int"})
    .SetKernelFn(PD_KERNEL(PSAMaskBackward));


