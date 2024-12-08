# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDA REQUIRED)
enable_language(CUDA)

message (STATUS "Found cuda.")
message (STATUS "Include Path:" ${CUDA_INCLUDE_DIRS})
message (STATUS "Library Path:" ${CUDA_LIBRARIES})
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -Xcompiler -fopenmp -std=c++14 -Xptxas -O3 --use_fast_math --disable-warnings -lineinfo
    -gencode arch=compute_70,code=sm_70
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_80,code=sm_80" )
elseif(WIN32)
    set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /openmp -Xcompiler /std:c++14 -Xcompiler /Zc:__cplusplus -Xcompiler /FS --use_fast_math
    -gencode arch=compute_70,code=sm_70
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_80,code=sm_80" )
endif()

message (STATUS "CUDA_NVCC_FLAGS:" ${CUDA_NVCC_FLAGS})

set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)
set(Zstd ${PROJECT_SOURCE_DIR}/ThirdParty/zstd)

include_directories(${AnnService})
include_directories(${Zstd}/lib)
include_directories(${CUDA_INCLUDE_DIRS})

set(cuFileLib /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcufile.so)

# file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h ${AnnService}/inc/Core/*.cuh ${AnnService}/inc/Helper/*.h ${AnnService}/inc/Helper/*.cuh)
# file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Core/*.cu ${AnnService}/src/Helper/*.cpp ${AnnService}/src/Helper/*.cu)
file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h ${AnnService}/inc/Helper/*.h)
file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp)

message(STATUS "HDR_FILES is ${HDR_FILES}")

message(STATUS "SRC_FILES is ${SRC_FILES}")

list(REMOVE_ITEM HDR_FILES
    ${AnnService}/inc/Core/Common/DistanceUtils.h
    ${AnnService}/inc/Core/Common/SIMDUtils.h 
    ${AnnService}/inc/Core/Common/InstructionUtils.h
    ${AnnService}/inc/Core/Common/CommonUtils.h
    )

list(REMOVE_ITEM SRC_FILES
    ${AnnService}/src/Core/Common/DistanceUtils.cpp
    ${AnnService}/src/Core/Common/SIMDUtils.cpp
    ${AnnService}/src/Core/Common/InstructionUtils.cpp
    )

file(GLOB_RECURSE GPU_HDR_FILES ${AnnService}/inc/Core/*.cuh ${AnnService}/inc/Helper/*.cuh)
file(GLOB_RECURSE GPU_SRC_FILES ${AnnService}/src/Helper/*.cu)

set_source_files_properties(${GPU_SRC_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(Definition -DGPUSPTAG -DDEBUG)
else()
    set(Definition -DGPUSPTAG)
endif()

add_library (DistanceUtils STATIC 
    inc/Core/Common/DistanceUtils.h
    inc/Core/Common/SIMDUtils.h
    inc/Core/Common/InstructionUtils.h
    inc/Core/Common/CommonUtils.h
    src/Core/Common/DistanceUtils.cpp
    src/Core/Common/SIMDUtils.cpp
    src/Core/Common/InstructionUtils.cpp
    )


if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(DistanceUtils PRIVATE -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -fPIC)
endif()

message(STATUS "GPU_HDR_FILES:${GPU_HDR_FILES}")
message(STATUS "GPU_SRC_FILES:${GPU_SRC_FILES}")

add_library(GPU_Lib SHARED ${GPU_HDR_FILES} ${GPU_SRC_FILES})
target_link_libraries (GPU_Lib DistanceUtils libzstd_shared ${NUMA_LIBRARY} ${CUDA_LIBRARIES} ${cuFileLib})
target_compile_definitions(GPU_Lib PRIVATE ${Definition})

add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLib DistanceUtils libzstd_shared GPU_Lib ${NUMA_LIBRARY} ${CUDA_LIBRARIES} ${cuFileLib})
target_compile_definitions(SPTAGLib PRIVATE ${Definition})

add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLibStatic DistanceUtils libzstd_static GPU_Lib ${NUMA_LIBRARY_STATIC} ${CUDA_LIBRARIES} ${cuFileLib})
target_compile_definitions(SPTAGLibStatic PRIVATE ${Definition})

# 👇 NOTE
# add_dependencies(SPTAGLibStatic SPTAGLib)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(SPTAGLibStatic PRIVATE -fPIC)
endif()

install(TARGETS SPTAGLib SPTAGLibStatic
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

if (NOT LIBRARYONLY)
    file(GLOB SERVER_HDR_FILES ${AnnService}/inc/Server/*.h ${AnnService}/inc/Socket/*.h)
    file(GLOB SERVER_FILES ${AnnService}/src/Server/*.cpp ${AnnService}/src/Socket/*.cpp)
    add_executable (server ${SERVER_FILES} ${SERVER_HDR_FILES})
    target_link_libraries(server ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB CLIENT_HDR_FILES ${AnnService}/inc/Client/*.h ${AnnService}/inc/Socket/*.h)
    file(GLOB CLIENT_FILES ${AnnService}/src/Client/*.cpp ${AnnService}/src/Socket/*.cpp)
    add_executable (client ${CLIENT_FILES} ${CLIENT_HDR_FILES})
    target_link_libraries(client ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB AGG_HDR_FILES ${AnnService}/inc/Aggregator/*.h ${AnnService}/inc/Socket/*.h ${AnnService}/inc/Server/QueryParser.h)
    file(GLOB AGG_FILES ${AnnService}/src/Aggregator/*.cpp ${AnnService}/src/Socket/*.cpp ${AnnService}/src/Server/QueryParser.cpp)
    add_executable (aggregator ${AGG_FILES} ${AGG_HDR_FILES})
    target_link_libraries(aggregator ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB BUILDER_FILES ${AnnService}/src/IndexBuilder/*.cpp)
    add_executable (indexbuilder ${BUILDER_FILES})
    target_link_libraries(indexbuilder ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB SEARCHER_FILES ${AnnService}/src/IndexSearcher/*.cpp)
    add_executable (indexsearcher ${SEARCHER_FILES})
    target_link_libraries(indexsearcher ${Boost_LIBRARIES} SPTAGLibStatic)
    
    file(GLOB QUANTIZER_HDR_FILES ${AnnService}/inc/Quantizer/*.h)
    file(GLOB QUANTIZER_FILES ${AnnService}/src/Quantizer/*.cpp)
    add_executable (quantizer ${QUANTIZER_FILES} ${QUANTIZER_HDR_FILES})
    target_link_libraries(quantizer ${Boost_LIBRARIES} SPTAGLibStatic)

    install(TARGETS server client aggregator indexbuilder indexsearcher quantizer
      RUNTIME DESTINATION bin
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib)
endif()

file(GLOB_RECURSE SSD_SERVING_HDR_FILES ${AnnService}/inc/SSDServing/*.h)
file(GLOB_RECURSE SSD_SERVING_FILES ${AnnService}/src/SSDServing/*.cpp)

add_executable(ssdserving ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdserving SPTAGLibStatic ${Boost_LIBRARIES})
target_compile_definitions(ssdserving PRIVATE _exe)

# for Test
add_library(ssdservingLib ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdservingLib SPTAGLibStatic ${Boost_LIBRARIES})

find_package(MPI)
if (MPI_FOUND)
    message (STATUS "Found MPI.")
    message (STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")
    message (STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
    include_directories (${MPI_CXX_INCLUDE_PATH})
    file(GLOB PARTITION_FILES ${AnnService}/src/BalancedDataPartition/*.cpp)
    add_executable(balanceddatapartition ${PARTITION_FILES})
    target_link_libraries(balanceddatapartition SPTAGLibStatic ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

install(TARGETS ssdservingLib ssdserving
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
