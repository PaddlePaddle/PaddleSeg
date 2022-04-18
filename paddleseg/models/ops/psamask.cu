#include "paddle/extension.h"

#include <vector>

#define CHECK_GPU_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif


template <typename data_t>
__global__ void psamask_collect_cuda_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             const int nthreads,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num_ * feature_H_ * feature_W_ * feature_H_ * feature_W_; i += blockDim.x * gridDim.x) {
        out_data[i] = 0;
    }
    for (int index{blockIdx.x * blockDim.x + threadIdx.x}; index< nthreads; index+=blockDim.x * gridDim.x) {
        const int w{index % feature_W_};
        const int h{index / feature_W_ % feature_H_};
        const int n{index / feature_W_ / feature_H_};
        const int hstart = max(0, half_mask_H_ - h);
        const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
        const int wstart = max(0, half_mask_W_ - w);
        const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
        for (int hidx{hstart}; hidx < hend; ++hidx){
            for (int widx{wstart}; widx < wend; ++widx) {
                out_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w] = x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
            }
        }
    }
}


template <typename data_t>
__global__ void psamask_distribute_cuda_forward_kernel(const data_t* x_data,
                             data_t* out_data,
                             const int nthreads,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num_ * feature_H_ * feature_W_ * feature_H_ * feature_W_; i += blockDim.x * gridDim.x) {
        out_data[i] = 0;
    } 
    for (int index{blockIdx.x * blockDim.x + threadIdx.x}; index< nthreads; index+=blockDim.x * gridDim.x) {
        const int w{index % feature_W_};
        const int h{index / feature_W_ % feature_H_};
        const int n{index / feature_W_ / feature_H_};
        const int hstart = max(0, half_mask_H_ - h);
        const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
        const int wstart = max(0, half_mask_W_ - w);
        const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
        for (int hidx{hstart}; hidx < hend; ++hidx){
            for (int widx{wstart}; widx < wend; ++widx) {
                out_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)] = x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
            }
        }
    }
}


template <typename data_t>
__global__ void psamask_collect_cuda_backward_kernel(const data_t* grad_out_data, const data_t* out,
                            data_t* grad_x_data,
                            const int nthreads,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < num_ * mask_H_ * mask_W_ * feature_H_ * feature_W_; i += blockDim.x * gridDim.x) {
        grad_x_data[i] = 0;
    }                      
     for (int index{blockIdx.x * blockDim.x + threadIdx.x}; index < nthreads; index+=blockDim.x * gridDim.x) {
        const int w{index % feature_W_};
        const int h{index / feature_W_ % feature_H_};
        const int n{index / feature_W_ / feature_H_};
        const int hstart = max(0, half_mask_H_ - h);
        const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
        const int wstart = max(0, half_mask_W_ - w);
        const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
        for (int hidx{hstart}; hidx < hend; ++hidx){
            for (int widx{wstart}; widx < wend; ++widx) {
                grad_x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] = grad_out_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w];
            }
        }
    }
}


template <typename data_t>
__global__ void psamask_distribute_cuda_backward_kernel(const data_t* grad_out_data, const data_t* out,
                            data_t* grad_x_data,
                            const int nthreads,
                             const int num_, const int feature_H_, const int feature_W_,
                             const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
     int gid = blockIdx.x * blockDim.x + threadIdx.x;
     for (int i = gid; i < num_ * mask_H_ * mask_W_ * feature_H_ * feature_W_; i += blockDim.x * gridDim.x) {
         grad_x_data[i] = 0;
     }        
     for (int index{blockIdx.x * blockDim.x + threadIdx.x}; index< nthreads; index+=blockDim.x * gridDim.x) {
        const int w{index % feature_W_};
        const int h{index / feature_W_ % feature_H_};
        const int n{index / feature_W_ / feature_H_};
        const int hstart = max(0, half_mask_H_ - h);
        const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
        const int wstart = max(0, half_mask_W_ - w);
        const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
        for (int hidx{hstart}; hidx < hend; ++hidx){
            for (int widx{wstart}; widx < wend; ++widx) {
                grad_x_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] = grad_out_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)];
            }
        }
    }
}


std::vector<paddle::Tensor> PSAMaskCUDAForward(const paddle::Tensor& x,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  CHECK_GPU_INPUT(x);
  

  auto out = paddle::Tensor(paddle::PlaceType::kGPU, std::vector<int64_t>{num_, feature_H_ * feature_W_, feature_H_, feature_W_});
  int numel = out.size();
  int nthreads = num_ * feature_H_ * feature_W_;
  int block = 512;
  if (psa_type == 0) {
    PD_DISPATCH_FLOATING_TYPES(
        x.type(), "psamask_collect_cuda_forward_kernel", ([&] {psamask_collect_cuda_forward_kernel<data_t><<<nthreads, block, 0, x.stream()>>>(x.data<data_t>(), out.mutable_data<data_t>(x.place()), nthreads, num_, feature_H_, feature_W_,mask_H_, mask_W_, half_mask_H_, half_mask_W_);}));
  }
  else{
      PD_DISPATCH_FLOATING_TYPES(
        x.type(), "psamask_distribute_cuda_forward_kernel", ([&] {
            psamask_distribute_cuda_forward_kernel<data_t><<<nthreads, block, 0, x.stream()>>>(
                x.data<data_t>(),  out.mutable_data<data_t>(x.place()), nthreads, num_, feature_H_, feature_W_,
                            mask_H_, mask_W_, half_mask_H_, half_mask_W_);
        }));
  }

  return {out};
}


std::vector<paddle::Tensor> PSAMaskCUDABackward(const paddle::Tensor& x,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out,
    const int psa_type, const int num_, const int feature_H_, const int feature_W_, 
    const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_) {
  CHECK_GPU_INPUT(x);
  CHECK_GPU_INPUT(out);
  CHECK_GPU_INPUT(grad_out);


  auto grad_x = paddle::Tensor(paddle::PlaceType::kGPU, x.shape());
  int numel = x.size();
  int nthreads = num_ * feature_H_ * feature_W_;
  int block = 512;
  if (psa_type == 0) {
    PD_DISPATCH_FLOATING_TYPES(out.type(), "psamask_collect_cuda_backward_kernel", ([&] {
                                psamask_collect_cuda_backward_kernel<data_t><<<nthreads,block, 0, x.stream()>>>(
                                    grad_out.data<data_t>(),
                                    out.data<data_t>(),
                                    grad_x.mutable_data<data_t>(x.place()),
                                    nthreads,
                                    num_, feature_H_, feature_W_,
                             mask_H_, mask_W_, half_mask_H_, half_mask_W_);
                                }));
  } else {
    PD_DISPATCH_FLOATING_TYPES(out.type(), "psamask_distribute_cuda_backward_kernel", ([&] {
                                psamask_distribute_cuda_backward_kernel<data_t><<<nthreads, block, 0, x.stream()>>>(
                                    grad_out.data<data_t>(),
                                    out.data<data_t>(),
                                    grad_x.mutable_data<data_t>(x.place()),
                                    nthreads,
                                    num_, feature_H_, feature_W_,
                             mask_H_, mask_W_, half_mask_H_, half_mask_W_);
                                }));
  }

  return {grad_x};
}