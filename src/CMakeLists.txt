cmake_minimum_required(VERSION 2.8.12)

project(brr)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

include(CheckCXXCompilerFlag)

# Check for -march=native support in the compiler
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Silence some annoying warnings form Eigen with gcc >= 6.1
SET(GCC_EIGEN_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-deprecated-declarations -DEIGEN_NO_DEBUG")
SET(CMAKE_CXX_FLAGS "-msse2 ${CMAKE_CXX_FLAGS} ${GCC_EIGEN_COMPILE_FLAGS}")

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

add_executable(${PROJECT_NAME}
    main.cpp
    BayesRRhp.cpp
    BayesRRpp.cpp
    BayesRRm.cpp
    BayesRMmapToy.cpp
    BayesRRtoy.cpp
    data.cpp
    distributions_boost.cpp
    gadgets.cpp
    gctb.cpp
    mympi.cpp
    options.cpp
    
)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


#target_link_libraries(${PROJECT_NAME} Qt5::Core)
