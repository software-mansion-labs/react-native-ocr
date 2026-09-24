cmake_minimum_required(VERSION 3.13)
project(RnOcr)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)

set(CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")

# Passed by build.gradle.kts: the installed react-native-executorch package,
# whose postinstall provides the OpenCV static libraries and headers.
if(NOT DEFINED RNE_DIR)
  message(FATAL_ERROR "RNE_DIR is not set; build.gradle.kts resolves it from node")
endif()
set(RNE_LIBS_DIR "${RNE_DIR}/third-party/android/libs")
set(RNE_INCLUDE_DIR "${RNE_DIR}/third-party/include")

find_package(ReactAndroid REQUIRED CONFIG)

file(GLOB_RECURSE RNOCR_SOURCES ${CPP_DIR}/*.cpp)
list(FILTER RNOCR_SOURCES EXCLUDE REGEX "/tests/")

add_library(${CMAKE_PROJECT_NAME} SHARED
    ${RNOCR_SOURCES}
    cpp-adapter.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CPP_DIR})
# The opencv2 headers matching the static libraries below.
target_include_directories(${CMAKE_PROJECT_NAME} SYSTEM PRIVATE ${RNE_INCLUDE_DIR})

# ------- OpenCV (static, from react-native-executorch) -------
# Only core and imgproc are used. KleidiCV is OpenCV's Arm HAL, linked on
# arm64 exactly as react-native-executorch does.
set(OPENCV_LINK_LIBS
    "${RNE_LIBS_DIR}/opencv/${ANDROID_ABI}/libopencv_imgproc.a"
    "${RNE_LIBS_DIR}/opencv/${ANDROID_ABI}/libopencv_core.a"
)
if(ANDROID_ABI STREQUAL "arm64-v8a")
  list(APPEND OPENCV_LINK_LIBS
    "${RNE_LIBS_DIR}/opencv-third-party/${ANDROID_ABI}/libkleidicv_hal.a"
    "${RNE_LIBS_DIR}/opencv-third-party/${ANDROID_ABI}/libkleidicv_thread.a"
    "${RNE_LIBS_DIR}/opencv-third-party/${ANDROID_ABI}/libkleidicv.a"
  )
endif()

# The static OpenCV was built with the OpenMP parallel backend.
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -fopenmp -static-openmp)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ReactAndroid::jsi
    log
    ${OPENCV_LINK_LIBS}
    z
)
