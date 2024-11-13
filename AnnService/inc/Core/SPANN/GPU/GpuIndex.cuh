// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/SPANN/GPU/GpuExtraFullGraphSearcher.cuh"

#include "inc/Core/Common.h"
#include "inc/Core/VectorIndex.h"

#include "inc/Core/Common/BKTree.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/SIMDUtils.h"
#include "inc/Core/Common/WorkSpacePool.h"

#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/Labelset.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Helper/VectorSetReader.h"

#include "inc/Core/SPANN/GPU/GpuOptions.h"

#include <functional>
#include <memory>
#include <shared_mutex>

namespace SPTAG {

namespace Helper {
class IniReader;
}

namespace GpuSPANN {
template <typename T> class SPANNResultIterator;

template <typename T> class GpuIndex : public VectorIndex {
private:
  std::shared_ptr<VectorIndex> m_index;
  std::shared_ptr<std::uint64_t> m_vectorTranslateMap;
  std::unordered_map<std::string, std::string> m_headParameters;

  std::shared_ptr<GpuIExtraSearcher> m_extraSearcher;

  Options m_options;

  std::function<float(const T *, const T *, DimensionType)> m_fComputeDistance;
  int m_iBaseSquare;
  // std::make_unique<SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace<HostMemoryPolicy>>>();
  std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>>
      m_workSpaceFactory;

public:
  GpuIndex() {
    m_workSpaceFactory = std::make_unique<
        SPTAG::COMMON::ThreadLocalWorkSpaceFactory<ExtraWorkSpace>>();
    m_fComputeDistance =
        std::function<float(const T *, const T *, DimensionType)>(
            COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod));
    m_iBaseSquare =
        (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
            ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>()
            : 1;
  }

  ~GpuIndex() {}

  inline std::shared_ptr<VectorIndex> GetMemoryIndex() { return m_index; }
  inline std::shared_ptr<GpuIExtraSearcher> GetDiskIndex() {
    return m_extraSearcher;
  }
  inline Options *GetOptions() { return &m_options; }

  inline SizeType GetNumSamples() const { return m_options.m_vectorSize; }
  inline DimensionType GetFeatureDim() const {
    return m_pQuantizer ? m_pQuantizer->ReconstructDim()
                        : m_index->GetFeatureDim();
  }

  inline int GetCurrMaxCheck() const { return m_options.m_maxCheck; }
  inline int GetNumThreads() const { return m_options.m_iSSDNumberOfThreads; }
  inline DistCalcMethod GetDistCalcMethod() const {
    return m_options.m_distCalcMethod;
  }
  inline IndexAlgoType GetIndexAlgoType() const { return IndexAlgoType::SPANN; }
  inline VectorValueType GetVectorValueType() const {
    return GetEnumValueType<T>();
  }

  void SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer);

  inline float AccurateDistance(const void *pX, const void *pY) const {
    if (m_options.m_distCalcMethod == DistCalcMethod::L2)
      return m_fComputeDistance((const T *)pX, (const T *)pY, m_options.m_dim);

    float xy = m_iBaseSquare - m_fComputeDistance((const T *)pX, (const T *)pY,
                                                  m_options.m_dim);
    float xx = m_iBaseSquare - m_fComputeDistance((const T *)pX, (const T *)pX,
                                                  m_options.m_dim);
    float yy = m_iBaseSquare - m_fComputeDistance((const T *)pY, (const T *)pY,
                                                  m_options.m_dim);
    return 1.0f - xy / (sqrt(xx) * sqrt(yy));
  }
  inline float ComputeDistance(const void *pX, const void *pY) const {
    return m_fComputeDistance((const T *)pX, (const T *)pY, m_options.m_dim);
  }
  inline float GetDistance(const void *target, const SizeType idx) const {
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                 "GetDistance NOT SUPPORT FOR SPANN");
    return -1;
  }
  inline bool ContainSample(const SizeType idx) const {
    return idx < m_options.m_vectorSize;
  }

  std::shared_ptr<std::vector<std::uint64_t>> BufferSize() const {
    std::shared_ptr<std::vector<std::uint64_t>> buffersize(
        new std::vector<std::uint64_t>);
    auto headIndexBufferSize = m_index->BufferSize();
    buffersize->insert(buffersize->end(), headIndexBufferSize->begin(),
                       headIndexBufferSize->end());
    buffersize->push_back(sizeof(long long) * m_index->GetNumSamples());
    return std::move(buffersize);
  }

  std::shared_ptr<std::vector<std::string>> GetIndexFiles() const {
    std::shared_ptr<std::vector<std::string>> files(
        new std::vector<std::string>);
    auto headfiles = m_index->GetIndexFiles();
    for (auto file : *headfiles) {
      files->push_back(m_options.m_headIndexFolder + FolderSep + file);
    }
    files->push_back(m_options.m_headIDFile);
    return std::move(files);
  }

  ErrorCode SaveConfig(std::shared_ptr<Helper::DiskIO> p_configout);
  ErrorCode SaveIndexData(
      const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams);

  ErrorCode LoadConfig(Helper::IniReader &p_reader);
  ErrorCode LoadIndexData(
      const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams);
  ErrorCode LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs);

  ErrorCode BuildIndex(const void *p_data, SizeType p_vectorNum,
                       DimensionType p_dimension, bool p_normalized = false,
                       bool p_shareOwnership = false);
  ErrorCode BuildIndex(bool p_normalized = false);
  ErrorCode SearchIndex(QueryResult &p_query,
                        bool p_searchDeleted = false) const;

  ErrorCode SearchDiskIndex(QueryResult &p_query,
                            SearchStats *p_stats = nullptr) const;
  bool SearchDiskIndexIterative(
      QueryResult &p_headQuery, QueryResult &p_query,
      ExtraWorkSpace *extraWorkspace) const;
  ErrorCode
  DebugSearchDiskIndex(QueryResult &p_query, int p_subInternalResultNum,
                       int p_internalResultNum, SearchStats *p_stats = nullptr,
                       std::set<int> *truth = nullptr,
                       std::map<int, std::set<int>> *found = nullptr) const;
  ErrorCode UpdateIndex();

  ErrorCode SetParameter(const char *p_param, const char *p_value,
                         const char *p_section = nullptr);
  std::string GetParameter(const char *p_param,
                           const char *p_section = nullptr) const;

  inline const void *GetSample(const SizeType idx) const { return nullptr; }
  inline SizeType GetNumDeleted() const { return 0; }
  inline bool NeedRefine() const { return false; }

  ErrorCode RefineSearchIndex(QueryResult &p_query,
                              bool p_searchDeleted = false) const {
    return ErrorCode::Undefined;
  }
  ErrorCode SearchTree(QueryResult &p_query) const {
    return ErrorCode::Undefined;
  }

  virtual std::shared_ptr<ResultIterator>
  GetIterator(const void *p_target, bool p_searchDeleted = false) const;

  virtual ErrorCode SearchIndexIterativeNext(QueryResult &p_query,
                                             COMMON::WorkSpace *workSpace,
                                             int p_batch, int &resultCount,
                                             bool p_isFirst,
                                             bool p_searchDeleted) const;

  virtual ErrorCode
  SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> workSpace) const;

  virtual bool
  SearchIndexIterativeFromNeareast(QueryResult &p_query,
                                   COMMON::WorkSpace *p_space, bool p_isFirst,
                                   bool p_searchDeleted = false) const;

  virtual std::unique_ptr<COMMON::WorkSpace> RentWorkSpace(int batch) const;

  virtual ErrorCode
  SearchIndexWithFilter(QueryResult &p_query,
                        std::function<bool(const ByteArray &)> filterFunc,
                        int maxCheck = 0, bool p_searchDeleted = false) const;

  ErrorCode AddIndex(const void *p_data, SizeType p_vectorNum,
                     DimensionType p_dimension,
                     std::shared_ptr<MetadataSet> p_metadataSet,
                     bool p_withMetaIndex = false, bool p_normalized = false) {
    return ErrorCode::Undefined;
  }
  ErrorCode DeleteIndex(const void *p_vectors, SizeType p_vectorNum) {
    return ErrorCode::Undefined;
  }
  ErrorCode DeleteIndex(const SizeType &p_id) { return ErrorCode::Undefined; }
  ErrorCode RefineIndex(
      const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
      IAbortOperation *p_abort) {
    return ErrorCode::Undefined;
  }
  ErrorCode RefineIndex(std::shared_ptr<VectorIndex> &p_newIndex) {
    return ErrorCode::Undefined;
  }
  ErrorCode SetWorkSpaceFactory(
      std::unique_ptr<
          SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>>
          up_workSpaceFactory) {
    SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>
        *raw_generic_ptr = up_workSpaceFactory.release();
    if (!raw_generic_ptr)
      return ErrorCode::Fail;

    SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace> *raw_specialized_ptr =
        dynamic_cast<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace> *>(
            raw_generic_ptr);
    if (!raw_specialized_ptr) {
      // If it is of type SPTAG::COMMON::WorkSpace, we should pass on to child
      // index
      if (!m_index) {
        delete raw_generic_ptr;
        return ErrorCode::Fail;
      } else {
        return m_index->SetWorkSpaceFactory(
            std::unique_ptr<
                SPTAG::COMMON::IWorkSpaceFactory<SPTAG::COMMON::IWorkSpace>>(
                raw_generic_ptr));
      }

    } else {
      m_workSpaceFactory =
          std::unique_ptr<SPTAG::COMMON::IWorkSpaceFactory<ExtraWorkSpace>>(
              raw_specialized_ptr);
      return ErrorCode::Success;
    }
  }

  SizeType GetGlobalVID(SizeType vid) {
    return static_cast<SizeType>((m_vectorTranslateMap.get())[vid]);
  }

private:
  bool CheckHeadIndexType();
  void SelectHeadAdjustOptions(int p_vectorCount);
  int SelectHeadDynamicallyInternal(
      const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID,
      const Options &p_opts, std::vector<int> &p_selected);
  void SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree,
                             int p_vectorCount, std::vector<int> &p_selected);

  template <typename InternalDataType>
  bool SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader);

  ErrorCode
  BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader);
};
} // namespace GpuSPANN
} // namespace SPTAG
