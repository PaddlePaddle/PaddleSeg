/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <cuda.h>
#include <stdio.h>

namespace custom_atomic{ 

#define CUDA_ATOMIC_WRAPPER(op, T) \
  __device__ __forceinline__ T CudaAtomic##op(T *address, const T val)

#define USE_CUDA_ATOMIC(op, T) \
  CUDA_ATOMIC_WRAPPER(op, T) { return atomic##op(address, val); }

// Default thread count per block(or block size).
// TODO(typhoonzero): need to benchmark against setting this value
//                    to 1024.
constexpr int PADDLE_CUDA_NUM_THREADS = 512;

// For atomicAdd.
USE_CUDA_ATOMIC(Add, float);
USE_CUDA_ATOMIC(Add, int);
USE_CUDA_ATOMIC(Add, unsigned int);
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
USE_CUDA_ATOMIC(Add, unsigned long long int);  // NOLINT

CUDA_ATOMIC_WRAPPER(Add, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  return CudaAtomicAdd(
      reinterpret_cast<unsigned long long int *>(address),  // NOLINT
      static_cast<unsigned long long int>(val));            // NOLINT
}

#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
USE_CUDA_ATOMIC(Add, double);
#else
CUDA_ATOMIC_WRAPPER(Add, double) {
  unsigned long long int *address_as_ull =                  // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

// For atomicMax
USE_CUDA_ATOMIC(Max, int);
USE_CUDA_ATOMIC(Max, unsigned int);
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350)
USE_CUDA_ATOMIC(Max, unsigned long long int);  // NOLINT
#else
CUDA_ATOMIC_WRAPPER(Max, unsigned long long int) {  // NOLINT
  if (*address >= val) {
    return *address;
  }

  unsigned long long int old = *address, assumed;  // NOLINT

  do {
    assumed = old;
    if (assumed >= val) {
      break;
    }

    old = atomicCAS(address, assumed, val);
  } while (assumed != old);
}
#endif

CUDA_ATOMIC_WRAPPER(Max, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  long long int res = *address;  // NOLINT
  while (val > res) {
    long long int old = res;                                           // NOLINT
    res = (long long int)atomicCAS((unsigned long long int *)address,  // NOLINT
                                   (unsigned long long int)old,        // NOLINT
                                   (unsigned long long int)val);       // NOLINT
    if (res == old) {
      break;
    }
  }
  return res;
}

CUDA_ATOMIC_WRAPPER(Max, float) {
  if (*address >= val) {
    return *address;
  }

  int *const address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) >= val) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

CUDA_ATOMIC_WRAPPER(Max, double) {
  if (*address >= val) {
    return *address;
  }

  unsigned long long int *const address_as_ull =            // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) {
      break;
    }

    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

// For atomicMin
USE_CUDA_ATOMIC(Min, int);
USE_CUDA_ATOMIC(Min, unsigned int);
// CUDA API uses unsigned long long int, we cannot use uint64_t here.
// It because unsigned long long int is not necessarily uint64_t
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350)
USE_CUDA_ATOMIC(Min, unsigned long long int);  // NOLINT
#else
CUDA_ATOMIC_WRAPPER(Min, unsigned long long int) {  // NOLINT
  if (*address <= val) {
    return *address;
  }

  unsigned long long int old = *address, assumed;  // NOLINT

  do {
    assumed = old;
    if (assumed <= val) {
      break;
    }

    old = atomicCAS(address, assumed, val);
  } while (assumed != old);
}
#endif

CUDA_ATOMIC_WRAPPER(Min, int64_t) {
  // Here, we check long long int must be int64_t.
  static_assert(sizeof(int64_t) == sizeof(long long int),  // NOLINT
                "long long should be int64");
  long long int res = *address;  // NOLINT
  while (val < res) {
    long long int old = res;                                           // NOLINT
    res = (long long int)atomicCAS((unsigned long long int *)address,  // NOLINT
                                   (unsigned long long int)old,        // NOLINT
                                   (unsigned long long int)val);       // NOLINT
    if (res == old) {
      break;
    }
  }
  return res;
}

CUDA_ATOMIC_WRAPPER(Min, float) {
  if (*address <= val) {
    return *address;
  }

  int *const address_as_i = reinterpret_cast<int *>(address);
  int old = *address_as_i, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) <= val) {
      break;
    }

    old = atomicCAS(address_as_i, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

CUDA_ATOMIC_WRAPPER(Min, double) {
  if (*address <= val) {
    return *address;
  }

  unsigned long long int *const address_as_ull =            // NOLINT
      reinterpret_cast<unsigned long long int *>(address);  // NOLINT
  unsigned long long int old = *address_as_ull, assumed;    // NOLINT

  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) {
      break;
    }

    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

}  // namespace custom_atomic