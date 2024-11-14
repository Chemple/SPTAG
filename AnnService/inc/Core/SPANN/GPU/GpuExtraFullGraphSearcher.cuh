// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inc/Core/Common/SIMDUtils.h"
#include "inc/Core/SPANN/GPU/GpuIExtraSearcher.cuh"
#include "inc/Helper/GPU/GpuAsyncFileReader.cuh"

#include "inc/Core/Common.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Core/SPANN/GPU/GpuCompressor.h"
#include "inc/Helper/Logging.h"
#include "inc/Helper/VectorSetReader.h"

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>

namespace SPTAG {
namespace GpuSPANN {

static int count = 0;

static int cuFileReadAsyncCount = 0;

extern std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO;

struct Selection {
  std::string m_tmpfile;
  size_t m_totalsize;
  size_t m_start;
  size_t m_end;
  std::vector<Edge> m_selections;
  static EdgeCompare g_edgeComparer;

  Selection(size_t totalsize, std::string tmpdir)
      : m_tmpfile(tmpdir + FolderSep + "selection_tmp"), m_totalsize(totalsize),
        m_start(0), m_end(totalsize) {
    remove(m_tmpfile.c_str());
    m_selections.resize(totalsize);
  }

  ErrorCode SaveBatch() {
    auto f_out = f_createIO();
    if (f_out == nullptr ||
        !f_out->Initialize(
            m_tmpfile.c_str(),
            std::ios::out | std::ios::binary |
                (fileexists(m_tmpfile.c_str()) ? std::ios::in : 0))) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Cannot open %s to save selection for batching!\n",
                   m_tmpfile.c_str());
      return ErrorCode::FailedOpenFile;
    }
    if (f_out->WriteBinary(
            sizeof(Edge) * (m_end - m_start), (const char *)m_selections.data(),
            sizeof(Edge) * m_start) != sizeof(Edge) * (m_end - m_start)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot write to %s!\n",
                   m_tmpfile.c_str());
      return ErrorCode::DiskIOFail;
    }
    std::vector<Edge> batch_selection;
    m_selections.swap(batch_selection);
    m_start = m_end = 0;
    return ErrorCode::Success;
  }

  ErrorCode LoadBatch(size_t start, size_t end) {
    auto f_in = f_createIO();
    if (f_in == nullptr ||
        !f_in->Initialize(m_tmpfile.c_str(), std::ios::in | std::ios::binary)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Cannot open %s to load selection batch!\n",
                   m_tmpfile.c_str());
      return ErrorCode::FailedOpenFile;
    }

    size_t readsize = end - start;
    m_selections.resize(readsize);
    if (f_in->ReadBinary(readsize * sizeof(Edge), (char *)m_selections.data(),
                         start * sizeof(Edge)) != readsize * sizeof(Edge)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Cannot read from %s! start:%zu size:%zu\n",
                   m_tmpfile.c_str(), start, readsize);
      return ErrorCode::DiskIOFail;
    }
    m_start = start;
    m_end = end;
    return ErrorCode::Success;
  }

  size_t lower_bound(SizeType node) {
    auto ptr = std::lower_bound(m_selections.begin(), m_selections.end(), node,
                                g_edgeComparer);
    return m_start + (ptr - m_selections.begin());
  }

  Edge &operator[](size_t offset) {
    if (offset < m_start || offset >= m_end) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Error read offset in selections:%zu\n", offset);
    }
    return m_selections[offset - m_start];
  }
};

template <typename ValueType>
class GpuExtraFullGraphSearcher : public GpuIExtraSearcher {
public:
  GpuExtraFullGraphSearcher() {
    m_enableDeltaEncoding = false;
    m_enablePostingListRearrange = false;
    m_enableDataCompression = false;
    m_enableDictTraining = true;
  }

  virtual ~GpuExtraFullGraphSearcher() {}

  virtual bool LoadIndex(Options &p_opt) {
    m_extraFullGraphFile =
        p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
    std::string curFile = m_extraFullGraphFile;
    p_opt.m_searchPostingPageLimit = max(
        p_opt.m_searchPostingPageLimit,
        static_cast<int>((p_opt.m_postingVectorLimit *
                              (p_opt.m_dim * sizeof(ValueType) + sizeof(int)) +
                          PageSize - 1) /
                         PageSize));
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Load index with posting page limit:%d\n",
                 p_opt.m_searchPostingPageLimit);
    m_indexFiles.resize(1);
    m_indexFiles[0] =
        std::make_shared<Helper::GPU::GpuDirectDiskIO>(curFile.c_str());

    try {
      m_totalListCount +=
          LoadingHeadInfo(curFile, p_opt.m_searchPostingPageLimit, m_listInfos);
    } catch (std::exception &e) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Error occurs when loading HeadInfo:%s\n", e.what());
      return false;
    }

    m_oneContext = (m_indexFiles.size() == 1);

    m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
    m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
    m_enableDataCompression = p_opt.m_enableDataCompression;
    m_enableDictTraining = p_opt.m_enableDictTraining;

    if (m_enablePostingListRearrange)
      m_parsePosting =
          &GpuExtraFullGraphSearcher<ValueType>::ParsePostingListRearrange;
    else
      m_parsePosting = &GpuExtraFullGraphSearcher<ValueType>::ParsePostingList;
    if (m_enableDeltaEncoding)
      m_parseEncoding =
          &GpuExtraFullGraphSearcher<ValueType>::ParseDeltaEncoding;
    else
      m_parseEncoding = &GpuExtraFullGraphSearcher<ValueType>::ParseEncoding;

    m_listPerFile = static_cast<int>(
        (m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

#ifndef _MSC_VER
    Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif
    return true;
  }

  virtual void SearchIndex(ExtraWorkSpace *p_exWorkSpace,
                           QueryResult &p_queryResults,
                           std::shared_ptr<VectorIndex> p_index,
                           SearchStats *p_stats, std::set<int> *truth,
                           std::map<int, std::set<int>> *found) {
    const uint32_t postingListCount =
        static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

    COMMON::QueryResultSet<ValueType> &queryResults =
        *((COMMON::QueryResultSet<ValueType> *)&p_queryResults);

    int diskRead = 0;
    int diskIO = 0;
    int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
    int unprocessed = 0;
#endif

    count++;
    printf("the count is %d\n", count);

    for (uint32_t pi = 0; pi < postingListCount; ++pi) {
      auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
      ListInfo *listInfo = &(m_listInfos[curPostingID]);

#ifndef BATCH_READ
      Helper::DiskIO *indexFile = m_indexFiles[fileid].get();
#endif

      diskRead += listInfo->listPageCount;
      diskIO += 1;
      listElements += listInfo->listEleCount;

      size_t totalBytes =
          (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
      char *buffer = (char *)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ
      auto &request = p_exWorkSpace->m_gdirectRequests[pi];
      request.m_offset = listInfo->listOffset;
      request.m_max_size = totalBytes;
      request.m_dev_ptr = buffer;

      request.m_payload = (void *)listInfo;
      request.m_success = false;

#ifdef BATCH_READ // async batch read
#else             // async read
      request.m_callback = [&p_exWorkSpace, &request](bool success) {
        p_exWorkSpace->m_processIocp.push(&request);
      };

      ++unprocessed;
      if (!(indexFile->ReadFileAsync(request))) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
        unprocessed--;
      }
#endif
#else // sync read
      auto numRead =
          indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
      if (numRead != totalBytes) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "File %s read bytes, expected: %zu, acutal: %llu.\n",
                     m_extraFullGraphFile.c_str(), totalBytes, numRead);
        throw std::runtime_error("File read mismatch");
      }
      // decompress posting list
      char *p_postingListFullData = buffer + listInfo->pageOffset;
      if (m_enableDataCompression) {
        DecompressPosting();
      }

      ProcessPosting();
#endif
    }

#ifdef ASYNC_READ
#ifdef BATCH_READ
    assert(m_indexFiles.size() == 1);
    Helper::GPU::BatchReadFileAsync(m_indexFiles[0],
                                    p_exWorkSpace->m_gdirectRequests,
                                    postingListCount, m_vectorInfoSize);
#else
    while (unprocessed > 0) {
      Helper::AsyncReadRequest *request;
      if (!(p_exWorkSpace->m_processIocp.pop(request)))
        break;

      --unprocessed;
      char *buffer = request->m_buffer;
      ListInfo *listInfo = static_cast<ListInfo *>(request->m_payload);
      // decompress posting list
      char *p_postingListFullData = buffer + listInfo->pageOffset;
      if (m_enableDataCompression) {
        DecompressPosting();
      }

      ProcessPosting();
    }
#endif
#endif
    if (truth) {
      for (uint32_t pi = 0; pi < postingListCount; ++pi) {
        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

        ListInfo *listInfo = &(m_listInfos[curPostingID]);
        char *buffer = (char *)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

        char *p_postingListFullData = buffer + listInfo->pageOffset;
        if (m_enableDataCompression) {
          p_postingListFullData =
              (char *)p_exWorkSpace->m_decompressBuffer.GetBuffer();
          if (listInfo->listEleCount != 0) {
            try {
              m_pCompressor->Decompress(
                  buffer + listInfo->pageOffset, listInfo->listTotalBytes,
                  p_postingListFullData,
                  listInfo->listEleCount * m_vectorInfoSize,
                  m_enableDictTraining);
            } catch (std::runtime_error &err) {
              SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                           "Decompress postingList %d  failed! %s, \n",
                           curPostingID, err.what());
              continue;
            }
          }
        }

        for (size_t i = 0; i < listInfo->listEleCount; ++i) {
          uint64_t offsetVectorID =
              m_enablePostingListRearrange
                  ? (m_vectorInfoSize - sizeof(int)) * listInfo->listEleCount +
                        sizeof(int) * i
                  : m_vectorInfoSize * i;
          int vectorID = *(
              reinterpret_cast<int *>(p_postingListFullData + offsetVectorID));
          if (truth && truth->count(vectorID))
            (*found)[curPostingID].insert(vectorID);
        }
      }
    }

    if (p_stats) {
      p_stats->m_totalListElementsCount = listElements;
      p_stats->m_diskIOCount = diskIO;
      p_stats->m_diskAccessCount = diskRead;
    }
  }

  virtual void SearchIndex_DEBUG(ExtraWorkSpace *p_exWorkSpace,
                                 QueryResult &p_queryResults,
                                 std::shared_ptr<VectorIndex> p_index,
                                 SearchStats *p_stats, std::set<int> *truth,
                                 std::map<int, std::set<int>> *found) {
    const uint32_t postingListCount =
        static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

    cuFileReadAsyncCount += postingListCount;

    // const uint32_t postingListCount = 2;
    // if (p_exWorkSpace->m_postingIDs.size() < 2) {
    //   SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "the size is too small");
    //   // exit(-1);
    // }

    // COMMON::QueryResultSet<ValueType> &queryResults =
    //     *((COMMON::QueryResultSet<ValueType> *)&p_queryResults);

    COMMON::QueryResultSet<float> &queryResults =
        *((COMMON::QueryResultSet<float> *)&p_queryResults);

    int diskRead = 0;
    int diskIO = 0;
    int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
    int unprocessed = 0;
#endif

    for (uint32_t pi = 0; pi < postingListCount; ++pi) {
      // cudaStreamSynchronize(p_exWorkSpace->m_cudaStreamPool.GetStream(pi));
      auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
      ListInfo *listInfo = &(m_listInfos[curPostingID]);

#ifndef BATCH_READ
      Helper::DiskIO *indexFile = m_indexFiles[fileid].get();
#endif

      diskRead += listInfo->listPageCount;
      diskIO += 1;
      listElements += listInfo->listEleCount;

      size_t totalBytes =
          (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
      char *buffer = (char *)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());
      cudaStream_t stream = p_exWorkSpace->m_cudaStreamPool.GetStream(pi);

#ifdef ASYNC_READ
      auto &request = p_exWorkSpace->m_gdirectRequests[pi];

      auto query = p_exWorkSpace->m_dev_query_vec;
      auto distance_tmp = p_exWorkSpace->m_dev_distance_tmp[pi];
      auto distance_res = p_exWorkSpace->m_dev_distance_res[pi];
      auto vid_tmp = p_exWorkSpace->m_dev_vid_tmp[pi];
      auto vid_res = p_exWorkSpace->m_dev_vid_res[pi];

      auto host_vid = p_exWorkSpace->m_host_vid_res[pi];
      auto host_distance = p_exWorkSpace->m_host_distance_res[pi];

      // cudaStreamSynchronize(stream);

      request.Initialize(buffer, totalBytes, listInfo->listOffset, 0, 0, stream,
                         (void *)listInfo, false, query, distance_tmp,
                         distance_res, vid_tmp, vid_res, host_vid,
                         host_distance);

#ifdef BATCH_READ // async batch read
#else             // async read
      request.m_callback = [&p_exWorkSpace, &request](bool success) {
        p_exWorkSpace->m_processIocp.push(&request);
      };

      ++unprocessed;
      if (!(indexFile->ReadFileAsync(request))) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file!\n");
        unprocessed--;
      }
#endif
#else // sync read
      auto numRead =
          indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
      if (numRead != totalBytes) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "File %s read bytes, expected: %zu, acutal: %llu.\n",
                     m_extraFullGraphFile.c_str(), totalBytes, numRead);
        throw std::runtime_error("File read mismatch");
      }
      // decompress posting list
      char *p_postingListFullData = buffer + listInfo->pageOffset;
      if (m_enableDataCompression) {
        DecompressPosting();
      }

      ProcessPosting();
#endif
    }

#ifdef ASYNC_READ
#ifdef BATCH_READ
    assert(m_indexFiles.size() == 1);
    // Helper::GPU::BatchReadFileAsync(m_indexFiles[0],
    //                                 p_exWorkSpace->m_gdirectRequests,
    //                                 postingListCount, m_vectorInfoSize);
    Helper::GPU::BatchReadFileSync(
        m_indexFiles[0], p_exWorkSpace->m_gdirectRequests, postingListCount,
        m_vectorInfoSize, queryResults, p_exWorkSpace->m_deduper);
#else
    while (unprocessed > 0) {
      Helper::AsyncReadRequest *request;
      if (!(p_exWorkSpace->m_processIocp.pop(request)))
        break;

      --unprocessed;
      char *buffer = request->m_buffer;
      ListInfo *listInfo = static_cast<ListInfo *>(request->m_payload);
      // decompress posting list
      char *p_postingListFullData = buffer + listInfo->pageOffset;
      if (m_enableDataCompression) {
        DecompressPosting();
      }

      ProcessPosting();
    }
#endif
#endif
    if (p_stats) {
      p_stats->m_totalListElementsCount = listElements;
      p_stats->m_diskIOCount = diskIO;
      p_stats->m_diskAccessCount = diskRead;
    }
    // queryResults.Debug();
  }

  virtual bool CheckValidPosting(SizeType postingID) {
    return m_listInfos[postingID].listEleCount != 0;
  }

private:
  struct ListInfo {
    std::size_t listTotalBytes = 0;

    int listEleCount = 0;

    std::uint16_t listPageCount = 0;

    std::uint64_t listOffset = 0;

    std::uint16_t pageOffset = 0;
  };

  int LoadingHeadInfo(const std::string &p_file, int p_postingPageLimit,
                      std::vector<ListInfo> &p_listInfos) {
    auto ptr = SPTAG::f_createIO();
    if (ptr == nullptr ||
        !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n",
                   p_file.c_str());
      throw std::runtime_error("Failed open file in LoadingHeadInfo");
    }
    m_pCompressor =
        std::make_unique<Compressor>(); // no need compress level to decompress

    int m_listCount;
    int m_totalDocumentCount;
    int m_listPageOffset;

    if (ptr->ReadBinary(sizeof(m_listCount),
                        reinterpret_cast<char *>(&m_listCount)) !=
        sizeof(m_listCount)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Failed to read head info file!\n");
      throw std::runtime_error("Failed read file in LoadingHeadInfo");
    }
    if (ptr->ReadBinary(sizeof(m_totalDocumentCount),
                        reinterpret_cast<char *>(&m_totalDocumentCount)) !=
        sizeof(m_totalDocumentCount)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Failed to read head info file!\n");
      throw std::runtime_error("Failed read file in LoadingHeadInfo");
    }
    if (ptr->ReadBinary(sizeof(m_iDataDimension),
                        reinterpret_cast<char *>(&m_iDataDimension)) !=
        sizeof(m_iDataDimension)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Failed to read head info file!\n");
      throw std::runtime_error("Failed read file in LoadingHeadInfo");
    }
    if (ptr->ReadBinary(sizeof(m_listPageOffset),
                        reinterpret_cast<char *>(&m_listPageOffset)) !=
        sizeof(m_listPageOffset)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Failed to read head info file!\n");
      throw std::runtime_error("Failed read file in LoadingHeadInfo");
    }

    if (m_vectorInfoSize == 0)
      m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
    else if (m_vectorInfoSize !=
             m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                   "Failed to read head info file! DataDimension and ValueType "
                   "are not match!\n");
      throw std::runtime_error(
          "DataDimension and ValueType don't match in LoadingHeadInfo");
    }

    size_t totalListCount = p_listInfos.size();
    p_listInfos.resize(totalListCount + m_listCount);

    size_t totalListElementCount = 0;

    std::map<int, int> pageCountDist;

    size_t biglistCount = 0;
    size_t biglistElementCount = 0;
    int pageNum;
    for (int i = 0; i < m_listCount; ++i) {
      ListInfo *listInfo = &(p_listInfos[totalListCount + i]);

      if (m_enableDataCompression) {
        if (ptr->ReadBinary(
                sizeof(listInfo->listTotalBytes),
                reinterpret_cast<char *>(&(listInfo->listTotalBytes))) !=
            sizeof(listInfo->listTotalBytes)) {
          SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                       "Failed to read head info file!\n");
          throw std::runtime_error("Failed read file in LoadingHeadInfo");
        }
      }
      if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char *>(&(
                                               pageNum))) != sizeof(pageNum)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      if (ptr->ReadBinary(sizeof(listInfo->pageOffset),
                          reinterpret_cast<char *>(&(listInfo->pageOffset))) !=
          sizeof(listInfo->pageOffset)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      if (ptr->ReadBinary(
              sizeof(listInfo->listEleCount),
              reinterpret_cast<char *>(&(listInfo->listEleCount))) !=
          sizeof(listInfo->listEleCount)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      if (ptr->ReadBinary(
              sizeof(listInfo->listPageCount),
              reinterpret_cast<char *>(&(listInfo->listPageCount))) !=
          sizeof(listInfo->listPageCount)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      listInfo->listOffset =
          (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
      if (!m_enableDataCompression) {
        listInfo->listTotalBytes = listInfo->listEleCount * m_vectorInfoSize;
        listInfo->listEleCount = min(
            listInfo->listEleCount,
            (min(static_cast<int>(listInfo->listPageCount), p_postingPageLimit)
             << PageSizeEx) /
                m_vectorInfoSize);
        listInfo->listPageCount = static_cast<std::uint16_t>(ceil(
            (m_vectorInfoSize * listInfo->listEleCount + listInfo->pageOffset) *
            1.0 / (1 << PageSizeEx)));
      }
      totalListElementCount += listInfo->listEleCount;
      int pageCount = listInfo->listPageCount;

      if (pageCount > 1) {
        ++biglistCount;
        biglistElementCount += listInfo->listEleCount;
      }

      if (pageCountDist.count(pageCount) == 0) {
        pageCountDist[pageCount] = 1;
      } else {
        pageCountDist[pageCount] += 1;
      }
    }

    if (m_enableDataCompression && m_enableDictTraining) {
      size_t dictBufferSize;
      if (ptr->ReadBinary(sizeof(size_t),
                          reinterpret_cast<char *>(&dictBufferSize)) !=
          sizeof(dictBufferSize)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      char *dictBuffer = new char[dictBufferSize];
      if (ptr->ReadBinary(dictBufferSize, dictBuffer) != dictBufferSize) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file!\n");
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      try {
        m_pCompressor->SetDictBuffer(std::string(dictBuffer, dictBufferSize));
      } catch (std::runtime_error &err) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to read head info file: %s \n", err.what());
        throw std::runtime_error("Failed read file in LoadingHeadInfo");
      }
      delete[] dictBuffer;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Finish reading header info, list count %d, total doc count "
                 "%d, dimension %d, list page offset %d.\n",
                 m_listCount, m_totalDocumentCount, m_iDataDimension,
                 m_listPageOffset);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Big page (>4K): list count %zu, total element count %zu.\n",
                 biglistCount, biglistElementCount);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n",
                 totalListElementCount);

    for (auto &ele : pageCountDist) {
      SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n",
                   ele.first, ele.second);
    }

    return m_listCount;
  }

  inline void ParsePostingListRearrange(uint64_t &offsetVectorID,
                                        uint64_t &offsetVector, int i,
                                        int eleCount) {
    offsetVectorID =
        (m_vectorInfoSize - sizeof(int)) * eleCount + sizeof(int) * i;
    offsetVector = (m_vectorInfoSize - sizeof(int)) * i;
  }

  inline void ParsePostingList(uint64_t &offsetVectorID, uint64_t &offsetVector,
                               int i, int eleCount) {
    offsetVectorID = m_vectorInfoSize * i;
    offsetVector = offsetVectorID + sizeof(int);
  }

  inline void ParseDeltaEncoding(std::shared_ptr<VectorIndex> &p_index,
                                 ListInfo *p_info, ValueType *vector) {
    ValueType *headVector = (ValueType *)p_index->GetSample(
        (SizeType)(p_info - m_listInfos.data()));
    COMMON::SIMDUtils::ComputeSum(vector, headVector, m_iDataDimension);
  }

  inline void ParseEncoding(std::shared_ptr<VectorIndex> &p_index,
                            ListInfo *p_info, ValueType *vector) {}

  void SelectPostingOffset(const std::vector<size_t> &p_postingListBytes,
                           std::unique_ptr<int[]> &p_postPageNum,
                           std::unique_ptr<std::uint16_t[]> &p_postPageOffset,
                           std::vector<int> &p_postingOrderInIndex) {
    p_postPageNum.reset(new int[p_postingListBytes.size()]);
    p_postPageOffset.reset(new std::uint16_t[p_postingListBytes.size()]);

    struct PageModWithID {
      int id;

      std::uint16_t rest;
    };

    struct PageModeWithIDCmp {
      bool operator()(const PageModWithID &a, const PageModWithID &b) const {
        return a.rest == b.rest ? a.id < b.id : a.rest > b.rest;
      }
    };

    std::set<PageModWithID, PageModeWithIDCmp> listRestSize;

    p_postingOrderInIndex.clear();
    p_postingOrderInIndex.reserve(p_postingListBytes.size());

    PageModWithID listInfo;
    for (size_t i = 0; i < p_postingListBytes.size(); ++i) {
      if (p_postingListBytes[i] == 0) {
        continue;
      }

      listInfo.id = static_cast<int>(i);
      listInfo.rest =
          static_cast<std::uint16_t>(p_postingListBytes[i] % PageSize);

      listRestSize.insert(listInfo);
    }

    listInfo.id = -1;

    int currPageNum = 0;
    std::uint16_t currOffset = 0;

    while (!listRestSize.empty()) {
      listInfo.rest = PageSize - currOffset;
      auto iter = listRestSize.lower_bound(listInfo); // avoid page-crossing
      if (iter == listRestSize.end() ||
          (listInfo.rest != PageSize && iter->rest == 0)) {
        ++currPageNum;
        currOffset = 0;
      } else {
        p_postPageNum[iter->id] = currPageNum;
        p_postPageOffset[iter->id] = currOffset;

        p_postingOrderInIndex.push_back(iter->id);

        currOffset += iter->rest;
        if (currOffset > PageSize) {
          SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Crossing extra pages\n");
          throw std::runtime_error("Read too many pages");
        }

        if (currOffset == PageSize) {
          ++currPageNum;
          currOffset = 0;
        }

        currPageNum +=
            static_cast<int>(p_postingListBytes[iter->id] / PageSize);

        listRestSize.erase(iter);
      }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "TotalPageNumbers: %d, IndexSize: %llu\n", currPageNum,
                 static_cast<uint64_t>(currPageNum) * PageSize + currOffset);
  }

private:
  std::string m_extraFullGraphFile;

  std::vector<ListInfo> m_listInfos;
  // NOTE(shiwen): what is the meaning? always true in ssdserving.?
  bool m_oneContext;

  std::vector<std::shared_ptr<Helper::GPU::GpuDirectDiskIO>> m_indexFiles;
  std::unique_ptr<Compressor> m_pCompressor;
  bool m_enableDeltaEncoding;
  bool m_enablePostingListRearrange;
  bool m_enableDataCompression;
  bool m_enableDictTraining;

  void (GpuExtraFullGraphSearcher<ValueType>::*m_parsePosting)(uint64_t &,
                                                               uint64_t &, int,
                                                               int);
  void (GpuExtraFullGraphSearcher<ValueType>::*m_parseEncoding)(
      std::shared_ptr<VectorIndex> &, ListInfo *, ValueType *);

  int m_vectorInfoSize = 0;
  int m_iDataDimension = 0;

  int m_totalListCount = 0;

  int m_listPerFile = 0;
};

} // namespace GpuSPANN
} // namespace SPTAG
