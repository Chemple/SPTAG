#pragma once

#include <cstddef>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string.h>

#include <cuda_runtime.h>
#include <cufile.h>
#include <sys/types.h>
namespace SPTAG {
namespace Helper {

struct GpuAsyncReadRequest {
  char *m_dev_ptr{nullptr};
  size_t m_max_size{0};
  off_t m_offset{0};
  off_t m_buf_off{0};
  ssize_t m_read_bytes_done{0};

  cudaStream_t m_cuda_stream;

  void *m_payload;
  bool m_success;

  char *m_dev_query;
  char *m_dev_distance_tmp;
  char *m_dev_distance_res;
  char *m_dev_vid_tmp;
  char *m_dev_vid_res;

  char *m_host_vid_res;
  char *m_host_distance_res;

  auto Initialize(char *m_dev_ptr, size_t m_max_size, off_t m_offset,
                  off_t m_buffer_off, ssize_t m_read_bytes_done,
                  cudaStream_t m_cuda_stream, void *m_payload, bool m_success,
                  char *dev_query, char *distance_tmp, char *distance_res,
                  char *vid_tmp, char *vid_res, char *host_vid_res,
                  char *host_distance_res) {
    this->m_dev_ptr = m_dev_ptr;
    this->m_max_size = m_max_size;
    this->m_offset = m_offset;
    this->m_buf_off = m_buffer_off;
    this->m_read_bytes_done = m_read_bytes_done;
    this->m_cuda_stream = m_cuda_stream;
    this->m_payload = m_payload;
    this->m_success = m_success;

    this->m_dev_query = dev_query;
    this->m_dev_distance_tmp = distance_tmp;
    this->m_dev_distance_res = distance_res;
    this->m_dev_vid_tmp = vid_tmp;
    this->m_dev_vid_res = vid_res;

    this->m_host_distance_res = host_distance_res;
    this->m_host_vid_res = host_vid_res;
  }
};
} // namespace Helper
} // namespace SPTAG