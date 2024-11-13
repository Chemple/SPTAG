#pragma once

#include "inc/Core/Common.h"
#include "inc/Core/SearchQuery.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/GPU/CudaStreamHelper.cuh"
#include "inc/Helper/GPU/CudaUtils.cuh"
#include "inc/Helper/GPU/GpuRequest.cuh"

#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/GPU/GpuOptions.h"
#include "inc/Helper/AsyncFileReader.h"
#include "inc/Helper/Logging.h"
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace SPTAG {
namespace GpuSPANN {

static int max_page_size = 0;

struct SearchStats {
  SearchStats()
      : m_check(0), m_exCheck(0), m_totalListElementsCount(0), m_diskIOCount(0),
        m_diskAccessCount(0), m_totalSearchLatency(0), m_totalLatency(0),
        m_exLatency(0), m_asyncLatency0(0), m_asyncLatency1(0),
        m_asyncLatency2(0), m_queueLatency(0), m_sleepLatency(0) {}

  int m_check;

  int m_exCheck;

  int m_totalListElementsCount;

  int m_diskIOCount;

  int m_diskAccessCount;

  double m_totalSearchLatency;

  double m_totalLatency;

  double m_exLatency;

  double m_asyncLatency0;

  double m_asyncLatency1;

  double m_asyncLatency2;

  double m_queueLatency;

  double m_sleepLatency;

  std::chrono::steady_clock::time_point m_searchRequestTime;

  int m_threadID;
};
template <typename T> class PageBuffer {
public:
  PageBuffer() : m_pageBufferSize(0) {}

  void ReservePageBuffer(std::size_t p_size) {
    if (m_pageBufferSize < p_size) {
      m_pageBufferSize = p_size;
      T *dev_ptr;
      CHKERR(cudaMalloc((void **)&dev_ptr, p_size * sizeof(T)));
      auto status = cuFileBufRegister(dev_ptr, p_size * sizeof(T), 0);
      if (status.err != CU_FILE_SUCCESS) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "cufile buffer register falied");
        exit(-1);
      }
      auto dealloc_func = [](T *ptr) {
        cuFileBufDeregister(ptr);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "dealloc one buffer slot\n");
        // TODO(shiwen): we should use memory pool later.
        CHKERR(cudaFree(ptr));
      };
      m_pageBuffer.reset(dev_ptr, dealloc_func);
    }
  }

  T *GetBuffer() { return m_pageBuffer.get(); }

  std::size_t GetPageSize() { return m_pageBufferSize; }

private:
  std::shared_ptr<T> m_pageBuffer;

  std::size_t m_pageBufferSize;
};

struct ExtraWorkSpace : public SPTAG::COMMON::IWorkSpace {
  ExtraWorkSpace() {}

  ~ExtraWorkSpace() { g_spaceCount--; }

  ExtraWorkSpace(ExtraWorkSpace &other) {
    Initialize(other.m_deduper.MaxCheck(), other.m_deduper.HashTableExponent(),
               (int)other.m_pageBuffers.size(),
               (int)(other.m_pageBuffers[0].GetPageSize()),
               other.m_enableDataCompression);
  }

  void Initialize(int p_maxCheck, int p_hashExp, int p_internalResultNum,
                  int p_maxPages, bool enableDataCompression) {
    m_postingIDs.reserve(p_internalResultNum);
    m_deduper.Init(p_maxCheck, p_hashExp);
    m_processIocp.reset(p_internalResultNum);
    m_pageBuffers.resize(p_internalResultNum);
    // TODO(shiwen): cudaSteamPool
    m_cudaStreamPool.Initialisze(p_internalResultNum);
    // NOTE(shiwen): async cudaMalloc.
    for (int pi = 0; pi < p_internalResultNum; pi++) {
      m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
    }
    m_gdirectRequests.resize(p_internalResultNum);
    m_enableDataCompression = enableDataCompression;
    if (enableDataCompression) {
      m_decompressBuffer.ReservePageBuffer(p_maxPages);
    }

    m_spaceID = g_spaceCount++;
    m_relaxedMono = false;

    cudaMalloc(&m_dev_query_vec, DIM * VALUE_SIZE);

    m_dev_distance_tmp.resize(p_internalResultNum);
    m_dev_distance_res.resize(p_internalResultNum);
    m_dev_vid_tmp.resize(p_internalResultNum);
    m_dev_vid_res.resize(p_internalResultNum);

    m_host_distance_res.resize(p_internalResultNum);
    m_host_vid_res.resize(p_internalResultNum);

    for (int pi = 0; pi < p_internalResultNum; pi++) {
      auto &dis_tmp = m_dev_distance_tmp[pi];
      auto &dis_res = m_dev_distance_res[pi];
      auto &vid_tmp = m_dev_vid_tmp[pi];
      auto &vid_res = m_dev_vid_res[pi];

      // NOTE(shiwen): the max vector number in one postinglist is 129.
      // TODO(shiwen): more correct way...

      cudaMalloc(&dis_tmp, VALUE_SIZE * 129);
      cudaMalloc(&dis_res, VALUE_SIZE * 129);
      cudaMalloc(&vid_tmp, sizeof(uint32_t) * 129);
      cudaMalloc(&vid_res, sizeof(uint32_t) * 129);

      auto &host_distance_res = m_host_distance_res[pi];
      auto &host_vid_res = m_host_vid_res[pi];

      host_distance_res = (char *)malloc(VALUE_SIZE * 129);
      host_vid_res = (char *)malloc(sizeof(uint32_t) * 129);
    }
  }

  // TODO(shiwen): waste GPU global memory
  void InitializeMaxResource(int p_maxCheck, int p_hashExp,
                             int p_internalResultNum, int p_maxPages,
                             bool enableDataCompression) {
    Initialize(p_maxCheck, p_hashExp, 96, 257 * 4096, enableDataCompression);
  }

  void Initialize(va_list &arg) {
    int maxCheck = va_arg(arg, int);
    int hashExp = va_arg(arg, int);
    int internalResultNum = va_arg(arg, int);
    int maxPages = va_arg(arg, int);
    bool enableDataCompression = bool(va_arg(arg, int));
    Initialize(maxCheck, hashExp, internalResultNum, maxPages,
               enableDataCompression);
  }

  void Clear(int p_internalResultNum, int p_maxPages,
             bool enableDataCompression) {

    if (p_maxPages > max_page_size) {
      max_page_size = p_maxPages;
    }
    if (p_internalResultNum > m_pageBuffers.size()) {
#ifdef DEBUG
      assert(false);
      exit(-1);
#endif
      m_postingIDs.reserve(p_internalResultNum);
      m_processIocp.reset(p_internalResultNum);
      m_pageBuffers.resize(p_internalResultNum);
      for (int pi = 0; pi < p_internalResultNum; pi++) {
        m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
      }
      m_gdirectRequests.resize(p_internalResultNum);
    } else if (p_maxPages > m_pageBuffers[0].GetPageSize()) {
#ifdef DEBUG
      assert(false);
      exit(-1);
#endif
      for (int pi = 0; pi < m_pageBuffers.size(); pi++)
        m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
    }

    m_enableDataCompression = enableDataCompression;
    if (enableDataCompression) {
#ifdef DEBUG
      assert(false);
      exit(-1);
#endif
      m_decompressBuffer.ReservePageBuffer(p_maxPages);
    }
  }

  static void Reset() { g_spaceCount = 0; }

  void SetQuery(char *query_data) {
    cudaMemcpy(m_dev_query_vec, query_data, DIM * VALUE_SIZE,
               cudaMemcpyHostToDevice);
  }

  auto GetQuery() { return m_dev_query_vec; }

  std::vector<int> m_postingIDs;

  COMMON::OptHashPosVector m_deduper;

  Helper::RequestQueue m_processIocp;

  std::vector<PageBuffer<std::uint8_t>> m_pageBuffers;

  char *m_dev_query_vec;

  const int DIM = 1024;
  const int VALUE_SIZE = 4;

  std::vector<char *> m_dev_distance_tmp;

  std::vector<char *> m_dev_vid_tmp;

  std::vector<char *> m_dev_distance_res;

  std::vector<char *> m_dev_vid_res;

  std::vector<char *> m_host_vid_res;

  std::vector<char *> m_host_distance_res;

  Helper::CudaStreamPool_t m_cudaStreamPool;

  bool m_enableDataCompression;
  PageBuffer<std::uint8_t> m_decompressBuffer;

  std::vector<Helper::GpuAsyncReadRequest> m_gdirectRequests;

  int m_spaceID;

  uint32_t m_pi;

  int m_offset;

  bool m_loadPosting;

  bool m_relaxedMono;

  int m_loadedPostingNum;

  static std::atomic_int g_spaceCount;
};

class GpuIExtraSearcher {
public:
  GpuIExtraSearcher() {}

  virtual ~GpuIExtraSearcher() {}

  virtual bool LoadIndex(Options &p_options) = 0;

  virtual void SearchIndex(ExtraWorkSpace *p_exWorkSpace,
                           QueryResult &p_queryResults,
                           std::shared_ptr<VectorIndex> p_index,
                           SearchStats *p_stats, std::set<int> *truth = nullptr,
                           std::map<int, std::set<int>> *found = nullptr) = 0;

  virtual void
  SearchIndex_DEBUG(ExtraWorkSpace *p_exWorkSpace, QueryResult &p_queryResults,
                    std::shared_ptr<VectorIndex> p_index, SearchStats *p_stats,
                    std::set<int> *truth = nullptr,
                    std::map<int, std::set<int>> *found = nullptr) = 0;

  virtual bool CheckValidPosting(SizeType postingID) = 0;
};
} // namespace GpuSPANN
} // namespace SPTAG