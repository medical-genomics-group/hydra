cmake_minimum_required(VERSION 3.1.3)

project(brr)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

include(CheckCXXCompilerFlag)

# Check for -march=native support in the compiler
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Silence some annoying warnings form Eigen with gcc >= 6.1
SET(GCC_EIGEN_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-deprecated-declarations -DEIGEN_NO_DEBUG  -DEIGEN_UNROLLING_LIMIT=1000")

if(FAST)
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math  -funroll-loops ")
endif()


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_EIGEN_COMPILE_FLAGS}")

#conditional compilation if using parallelised or vectorised reductions
if(PARUP)
  add_definitions(-DPARUP)
endif()

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)


add_library(libbrr-dev
    BayesRRm.cpp
    BayesRRmz.cpp
    data.cpp
    distributions_boost.cpp
    gadgets.cpp
    options.cpp
    samplewriter.cpp
    compression.cpp
    limitsequencegraph.cpp
    analysisgraph.cpp
)

add_executable(${PROJECT_NAME}
    main.cpp
    BayesRRm.cpp
    BayesRRmz.cpp
    data.cpp
    distributions_boost.cpp
    gadgets.cpp
    options.cpp
    samplewriter.cpp
    compression.cpp
    limitsequencegraph.cpp
    analysisgraph.cpp
)

set_property(TARGET libbrr-dev PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET libbrr-dev PROPERTY CXX_STANDARD 17)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${ZLIB_LIBRARIES})
  target_link_libraries(libbrr-dev ${ZLIB_LIBRARIES})
endif(ZLIB_FOUND)

set(TBB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../tbb)
include(${TBB_ROOT}/cmake/TBBBuild.cmake)
tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS stdver=c++17 tbb_cpf=1)
find_package(TBB REQUIRED tbb_preview)

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} ${TBB_IMPORTED_TARGETS})
target_link_libraries(libbrr-dev ${MPI_LIBRARIES} ${TBB_IMPORTED_TARGETS})

if(MPI_COMPILE_FLAGS)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(libbrr-dev PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
    set_target_properties(libbrr-dev PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${PROJECT_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
    