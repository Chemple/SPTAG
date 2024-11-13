#pragma once

#include "GpuRequest.cuh"
#include "inc/Core/Common.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Helper/Logging.h"

#include <fcntl.h>
#include <iostream>
#include <memory>
#include <string.h>

#include <cuda_runtime.h>
#include <cufile.h>

#include <cstdint>
#include <vector>

namespace SPTAG {
namespace Helper {
namespace GPU {

class GpuDirectDiskIO {
public:
  GpuDirectDiskIO(const char *file_name) {
    auto fd = open(file_name, O_RDONLY | O_DIRECT);
    if (fd < 0) {
      std::cerr << "can not open file: " << file_name << "\n";
    }
    memset((void *)&m_cufile_descr, 0, sizeof(CUfileDescr_t));
    m_cufile_descr.handle.fd = fd;
    // NOTE(shiwen): Linux based fd
    m_cufile_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    auto status = cuFileHandleRegister(&m_cufile_handle, &m_cufile_descr);
    if (status.err != CU_FILE_SUCCESS) {
      std::cerr << "can not cudafile register file: " << file_name << "\n";
      exit(-1);
    }
  }

  ~GpuDirectDiskIO() {
    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "deallocate buffer\n");
    cuFileHandleDeregister(m_cufile_handle);
  }

  CUfileHandle_t m_cufile_handle;
  CUfileDescr_t m_cufile_descr;
};

void BatchReadFileAsync(
    std::shared_ptr<Helper::GPU::GpuDirectDiskIO> &gpu_diskio_handler,
    std::vector<GpuAsyncReadRequest> &readRequests, int num,
    int m_vectorInfoSize);

// NOTE(shiwen):only support float value first.
void BatchReadFileSync(
    std::shared_ptr<Helper::GPU::GpuDirectDiskIO> &gpu_diskio_handler,
    std::vector<GpuAsyncReadRequest> &readRequests, int num,
    int m_vectorInfoSize, COMMON::QueryResultSet<float> &queryResults,
    COMMON::OptHashPosVector &m_deduper);

} // namespace GPU
} // namespace Helper
} // namespace SPTAG