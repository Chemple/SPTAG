#include "inc/Core/Common.h"
#include "inc/Helper/Logging.h"

#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/GPU/GpuExtraFullGraphSearcher.cuh"
#include "inc/Helper/GPU/GpuAsyncFileReader.cuh"
#include "inc/Helper/GPU/GpuRequest.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sys/types.h>

namespace SPTAG {
namespace Helper {
namespace GPU {

const int DIM = 1024;

struct ListInfo {
  std::size_t listTotalBytes = 0;

  int listEleCount = 0;

  std::uint16_t listPageCount = 0;

  std::uint64_t listOffset = 0;

  std::uint16_t pageOffset = 0;
};

__global__ void PrintVID(const char *dev_postingListFullData,
                         const int list_elem_ct, const int m_vectorInfoSize) {
  int32_t elem_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t offset_vid = m_vectorInfoSize * elem_id;
  uint64_t offset_vector = offset_vid + sizeof(int32_t);
  const int32_t *vid =
      (reinterpret_cast<const int *>(dev_postingListFullData + offset_vid));
  if (elem_id < list_elem_ct) {
    if (vid == nullptr) {
      printf("vid is nullptr, list elem ct is%d\n", list_elem_ct);
      return;
    }
    if (*vid == 0) {
      printf("the vid is 0, bad...\n");
      return;
    }
    printf("the vid is %d\n", *vid);
  }
}

namespace cg = cooperative_groups;

template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <> struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

// each block calculate one vector of one postinglist.
template <typename T>
__global__ void
DistanceCalculate(const char *dev_postingListFullData,
                  const char *dev_query_vector, const int m_vectorInfoSize,
                  const int list_elem_ct, unsigned int n, unsigned int dim,
                  char *distance_result, char *vid_result) {

  // blockSize == dim
  assert(dim * list_elem_ct == n);

  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  unsigned int tid = threadIdx.x;
  unsigned int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t offset_vid = m_vectorInfoSize * blockIdx.x;
  uint64_t offset_vector = offset_vid + sizeof(int32_t);

  if (elem_id < n) {
    uint64_t element_loc_in_query = elem_id % dim;
    uint64_t element_value_offset =
        offset_vector + element_loc_in_query * sizeof(T);
    const T *element_value = (reinterpret_cast<const T *>(
        dev_postingListFullData + element_value_offset));
    T diff = (*element_value - ((T *)dev_query_vector)[tid]) *
             (*element_value - ((T *)dev_query_vector)[tid]);
    sdata[tid] = diff;
  } else {
    sdata[tid] = 0;
  }

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    cg::sync(cta);
  }

  if (tid == 0) {
    ((T *)distance_result)[blockIdx.x] = sdata[0];
    // TODO(shiwen): maybe bad performance, change the postinglist layout.
    const int32_t *vid =
        (reinterpret_cast<const int *>(dev_postingListFullData + offset_vid));
    ((int32_t *)vid_result)[blockIdx.x] = *vid;
  }
}

inline void LaunchDistanceComputeKernelAsync(
    u_int32_t blockSize, u_int32_t gridSize, cudaStream_t cudaStream,
    const char *dev_postingListFullData, const char *dev_query_vector,
    const int m_vectorInfoSize, const int list_elem_ct, unsigned int n,
    unsigned int dim, char *distance_result, char *vid_result) {
  DistanceCalculate<float>
      <<<gridSize, blockSize, 1024 * sizeof(float), cudaStream>>>(
          dev_postingListFullData, dev_query_vector, m_vectorInfoSize,
          list_elem_ct, n, dim, distance_result, vid_result);
}

inline void LaunchDistanceComputeKernelSync(
    u_int32_t blockSize, u_int32_t gridSize,
    const char *dev_postingListFullData, const char *dev_query_vector,
    const int m_vectorInfoSize, const int list_elem_ct, unsigned int n,
    unsigned int dim, char *distance_result, char *vid_result) {
  DistanceCalculate<float><<<gridSize, blockSize, 1024 * sizeof(float)>>>(
      dev_postingListFullData, dev_query_vector, m_vectorInfoSize, list_elem_ct,
      n, dim, distance_result, vid_result);
}

template <typename T>
__device__ void Comparator_v2(T &keyA, uint32_t &valA, T &keyB, uint32_t &valB,
                              uint dir) {
  // Swap if the direction does not match the ordering condition
  if ((keyA > keyB) == dir) {
    T tempKey = keyA;
    uint32_t tempVal = valA;
    keyA = keyB;
    valA = valB;
    keyB = tempKey;
    valB = tempVal;
  }
}

const int SHARED_SIZE_LIMIT = 256;

template <typename T>
__global__ void bitonicSortShared_v2(T *d_DstKey, uint32_t *d_DstVal,
                                     T *d_SrcKey, uint32_t *d_SrcVal,
                                     uint32_t arrayLength, uint dir) {
  // Thread block handle
  cg::thread_block cta = cg::this_thread_block();

  // Shared memory storage for keys and values
  __shared__ T s_key[SHARED_SIZE_LIMIT];
  __shared__ uint32_t s_val[SHARED_SIZE_LIMIT];

  // Offset to the beginning of the sub-batch
  unsigned int tid = threadIdx.x;
  unsigned int blockOffset = blockIdx.x * SHARED_SIZE_LIMIT;

  // Load data into shared memory, checking bounds to avoid overflow
  if (tid < arrayLength) {
    s_key[tid] = d_SrcKey[blockOffset + tid];
    s_val[tid] = d_SrcVal[blockOffset + tid];
  }
  if (tid + SHARED_SIZE_LIMIT / 2 < arrayLength) {
    s_key[tid + SHARED_SIZE_LIMIT / 2] =
        d_SrcKey[blockOffset + tid + SHARED_SIZE_LIMIT / 2];
    s_val[tid + SHARED_SIZE_LIMIT / 2] =
        d_SrcVal[blockOffset + tid + SHARED_SIZE_LIMIT / 2];
  }

  cg::sync(cta);

  // Bitonic sort
  for (uint size = 2; size <= arrayLength; size <<= 1) {
    // Determine the direction for this stage
    uint stageDir = dir ^ ((tid & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * tid - (tid & (stride - 1));
      if (pos + stride < SHARED_SIZE_LIMIT && pos < arrayLength) {
        Comparator_v2(s_key[pos], s_val[pos], s_key[pos + stride],
                      s_val[pos + stride], stageDir);
      }
    }
  }

  // Final merge step with fixed direction `dir`
  for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);
    uint pos = 2 * tid - (tid & (stride - 1));
    if (pos + stride < SHARED_SIZE_LIMIT && pos < arrayLength) {
      Comparator_v2(s_key[pos], s_val[pos], s_key[pos + stride],
                    s_val[pos + stride], dir);
    }
  }

  // Synchronize before writing results
  cg::sync(cta);

  // Write results from shared memory back to global memory
  if (tid < arrayLength) {
    d_DstKey[blockOffset + tid] = s_key[tid];
    d_DstVal[blockOffset + tid] = s_val[tid];
  }
  if (tid + SHARED_SIZE_LIMIT / 2 < arrayLength) {
    d_DstKey[blockOffset + tid + SHARED_SIZE_LIMIT / 2] =
        s_key[tid + SHARED_SIZE_LIMIT / 2];
    d_DstVal[blockOffset + tid + SHARED_SIZE_LIMIT / 2] =
        s_val[tid + SHARED_SIZE_LIMIT / 2];
  }
}

template <typename T>
inline void LaunchSortKernelAsync(u_int32_t blockSize, u_int32_t gridSize,
                                  cudaStream_t cudaStream, T *d_DstKey,
                                  uint32_t *d_DstVal, T *d_SrcKey,
                                  uint32_t *d_SrcVal, uint32_t arrayLength,
                                  uint dir) {
  bitonicSortShared_v2<float><<<gridSize, blockSize, 0, cudaStream>>>(
      d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
}

template <typename T>
inline void LaunchSortKernelSync(u_int32_t blockSize, u_int32_t gridSize,
                                 T *d_DstKey, uint32_t *d_DstVal, T *d_SrcKey,
                                 uint32_t *d_SrcVal, uint32_t arrayLength,
                                 uint dir) {
  bitonicSortShared_v2<float><<<gridSize, blockSize, 0>>>(
      d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
}

void LaunchPrintVIDKernel(int32_t blockSize, int32_t gridSize,
                          cudaStream_t stream, char *dev_postingListFullData,
                          const int list_elem_ct, const int m_vectorInfoSize) {
  // printf("total element number is %d\n", list_elem_ct);
  PrintVID<<<gridSize, blockSize, 0, stream>>>(dev_postingListFullData,
                                               list_elem_ct, m_vectorInfoSize);
}

void LaunchGpuKernel_DEBUG(GpuAsyncReadRequest &readRequest,
                           int m_vectorInfoSize,
                           COMMON::QueryResultSet<float> &queryResults,
                           COMMON::OptHashPosVector &m_deduper) {

  char *dev_buffer = (char *)readRequest.m_dev_ptr;
  ListInfo *listinfo = (ListInfo *)(readRequest.m_payload);

  char *dev_postingListFullData = dev_buffer + listinfo->pageOffset;

  cudaStream_t stream = readRequest.m_cuda_stream;

  auto query = readRequest.m_dev_query;
  auto distance_tmp = readRequest.m_dev_distance_tmp;
  auto distance_res = readRequest.m_dev_distance_res;
  auto vid_tmp = readRequest.m_dev_vid_tmp;
  auto vid_res = readRequest.m_dev_vid_res;

  auto cudaStream = readRequest.m_cuda_stream;

  auto list_count = listinfo->listEleCount;

  LaunchDistanceComputeKernelSync(1024, list_count, dev_postingListFullData,
                                  query, m_vectorInfoSize, list_count,
                                  DIM * list_count, DIM, distance_tmp, vid_tmp);

  // LaunchSortKernelSync<float>(128, 4, (float *)distance_res,
  //                             (uint32_t *)vid_res, (float *)distance_tmp,
  //                             (uint32_t *)vid_tmp, list_count, true);

  auto &host_vid_res = readRequest.m_host_vid_res;
  auto &host_distance_res = readRequest.m_host_distance_res;

  cudaMemcpy(host_vid_res, vid_tmp, list_count * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_distance_res, distance_tmp, list_count * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (auto i = 0; i < list_count; i++) {
    if (m_deduper.CheckAndSet(((int32_t *)host_vid_res)[i])) {
      continue;
    }
    queryResults.AddPoint(((int32_t *)host_vid_res)[i],
                          ((float *)host_distance_res)[i]);
  }
  // queryResults.Debug();
}

// NOTE(shiwen):only support float value first.
void BatchReadFileSync(
    std::shared_ptr<Helper::GPU::GpuDirectDiskIO> &gpu_diskio_handler,
    std::vector<GpuAsyncReadRequest> &readRequests, int num,
    int m_vectorInfoSize, COMMON::QueryResultSet<float> &queryResults,
    COMMON::OptHashPosVector &m_deduper) {
  for (auto i = 0; i < num; i++) {
    auto status = cuFileRead(
        gpu_diskio_handler->m_cufile_handle,
        (unsigned char *)readRequests[i].m_dev_ptr, readRequests[i].m_max_size,
        readRequests[i].m_offset, readRequests[i].m_buf_off);
    if (status <= 0) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "cuFileRead failed");
      exit(-1);
    }
    LaunchGpuKernel_DEBUG(readRequests[i], m_vectorInfoSize, queryResults,
                          m_deduper);
  }
}

void BatchReadFileAsync(
    std::shared_ptr<Helper::GPU::GpuDirectDiskIO> &gpu_diskio_handler,
    std::vector<GpuAsyncReadRequest> &readRequests, int num,
    int m_vectorInfoSize) {
  // cudaDeviceSynchronize();

  // NOTE(shiwen): gpu direct read
  for (auto i = 0; i < num; i++) {
    auto status = cuFileReadAsync(
        gpu_diskio_handler->m_cufile_handle,
        (unsigned char *)readRequests[i].m_dev_ptr,
        &(readRequests[i].m_max_size), &(readRequests[i].m_offset),
        &(readRequests[i].m_buf_off), &(readRequests[i].m_read_bytes_done),
        readRequests[i].m_cuda_stream);
    if (status.err != CU_FILE_SUCCESS) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "cuFileReadAsync failed");
      exit(-1);
    }
    // LaunchGpuKernel_DEBUG(readRequests[i], m_vectorInfoSize);
  }

  // NOTE(shiwen): lauch gpu kernel
}
} // namespace GPU
} // namespace Helper
} // namespace SPTAG