#pragma once

#include "inc/Helper/GPU/CudaUtils.cuh"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <list>
#include <vector>

namespace SPTAG {
namespace Helper {

// TODO(shiwen): "stream" safe streampool??
class CudaStreamPool {
public:
  using loc_t = uint32_t;
  uint32_t m_size{0};

  std::vector<cudaStream_t> m_cudaStreams{};

  auto Initialisze(uint32_t size) {
    m_cudaStreams.resize(size);
    for (auto i = 0; i < size; i++) {
      CHKERR(cudaStreamCreateWithFlags(&m_cudaStreams.data()[i],
                                cudaStreamNonBlocking));
    }
  }

  auto Resize(uint32_t resize) {
    assert(resize > m_size);
    m_size = resize;
    m_cudaStreams.resize(m_size);
  }

  auto GetStream(uint32_t loc) { return m_cudaStreams[loc]; }
};

using CudaStreamPool_t = CudaStreamPool;
} // namespace Helper
} // namespace SPTAG