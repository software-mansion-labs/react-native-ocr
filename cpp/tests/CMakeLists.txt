cmake_minimum_required(VERSION 3.24)
project(RnOcrTests CXX)

# Host-side unit tests for the kernels under cpp/ops/.
#
# The kernels are plain C++ over raw buffers and never touch JSI, so they build
# against whatever OpenCV the host has (Homebrew, apt) and need no JavaScript
# engine. The JSI bindings (cpp/ops/bindings.cpp, cpp/core/*) are exercised by
# the example app instead.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# --- OpenCV -----------------------------------------------------------------
# Only core and imgproc are used. Prefer the CMake package (Homebrew, a full
# distro OpenCV) and fall back to the two libraries directly: on Debian/Ubuntu
# OpenCVConfig.cmake ships only with the `libopencv-dev` meta-package, which
# drags in VTK and ~220 packages, whereas libopencv-{core,imgproc}-dev is small.
# OpenCV 5 moved minAreaRect into a separate `geometry` module, so it is linked
# when present.
add_library(rnocr_opencv INTERFACE)
find_package(OpenCV QUIET COMPONENTS core imgproc OPTIONAL_COMPONENTS geometry)
if(OpenCV_FOUND)
  message(STATUS "OpenCV: using CMake package ${OpenCV_VERSION}")
  target_link_libraries(rnocr_opencv INTERFACE ${OpenCV_LIBS})
  target_include_directories(rnocr_opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
else()
  find_path(OPENCV_INCLUDE_DIR NAMES opencv2/core.hpp PATH_SUFFIXES opencv4 opencv5)
  find_library(OPENCV_CORE_LIB NAMES opencv_core)
  find_library(OPENCV_IMGPROC_LIB NAMES opencv_imgproc)
  find_library(OPENCV_GEOMETRY_LIB NAMES opencv_geometry)
  if(NOT OPENCV_INCLUDE_DIR OR NOT OPENCV_CORE_LIB OR NOT OPENCV_IMGPROC_LIB)
    message(FATAL_ERROR
      "OpenCV (core + imgproc) not found. Install it with:\n"
      "  brew install opencv\n"
      "  apt-get install libopencv-core-dev libopencv-imgproc-dev")
  endif()
  message(STATUS "OpenCV: using ${OPENCV_CORE_LIB}")
  target_include_directories(rnocr_opencv INTERFACE "${OPENCV_INCLUDE_DIR}")
  target_link_libraries(rnocr_opencv INTERFACE "${OPENCV_CORE_LIB}" "${OPENCV_IMGPROC_LIB}")
  if(OPENCV_GEOMETRY_LIB)
    target_link_libraries(rnocr_opencv INTERFACE "${OPENCV_GEOMETRY_LIB}")
  endif()
endif()

# --- GoogleTest -------------------------------------------------------------
# An installed copy (brew install googletest, apt install libgtest-dev) is
# preferred; otherwise a pinned release is fetched at configure time.
find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
  message(STATUS "GoogleTest: using installed package ${GTest_DIR}")
else()
  message(STATUS "GoogleTest: not installed, fetching v1.17.0")
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# --- Kernels under test -----------------------------------------------------
add_library(rnocr_kernels STATIC
  "${CPP_DIR}/ops/image_ops.cpp"
  "${CPP_DIR}/ops/craft_ops.cpp")
target_include_directories(rnocr_kernels PUBLIC "${CPP_DIR}")
target_link_libraries(rnocr_kernels PUBLIC rnocr_opencv)
target_compile_options(rnocr_kernels PRIVATE -Wall -Wextra)

# --- JSI layer: compile check only ------------------------------------------
# The bindings and the core helpers need jsi.h. When the repository's
# node_modules is present (react-native ships the header), they are compiled
# into an object library that nothing links, so a type error in the JSI layer
# surfaces here instead of in a ten-minute Xcode or Gradle build. jsi.h is
# header-only for this purpose; the symbols it references live in the app.
set(JSI_INCLUDE_DIR "${CPP_DIR}/../node_modules/react-native/ReactCommon/jsi")
if(EXISTS "${JSI_INCLUDE_DIR}/jsi/jsi.h")
  message(STATUS "JSI headers: compile-checking the bindings against ${JSI_INCLUDE_DIR}")
  add_library(rnocr_jsi_syntax OBJECT
    "${CPP_DIR}/RnOcr.cpp"
    "${CPP_DIR}/core/buffers.cpp"
    "${CPP_DIR}/core/guard.cpp"
    "${CPP_DIR}/core/options.cpp"
    "${CPP_DIR}/core/utils.cpp"
    "${CPP_DIR}/ops/bindings.cpp")
  target_include_directories(rnocr_jsi_syntax PRIVATE "${CPP_DIR}")
  target_include_directories(rnocr_jsi_syntax SYSTEM PRIVATE "${JSI_INCLUDE_DIR}")
  target_link_libraries(rnocr_jsi_syntax PRIVATE rnocr_opencv)
  target_compile_options(rnocr_jsi_syntax PRIVATE -Wall -Wextra)
else()
  message(STATUS "JSI headers: node_modules not found, skipping the bindings compile check")
endif()

# --- Parity oracle ----------------------------------------------------------
# The react-native-executorch CRAFT decoder the kernel was ported from, built
# only for CraftParityTest to compare against on real heatmaps.
add_library(rnocr_reference_craft STATIC support/ReferenceCraft.cpp)
target_include_directories(rnocr_reference_craft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(rnocr_reference_craft PUBLIC rnocr_opencv)

# --- Suites -----------------------------------------------------------------
enable_testing()
include(GoogleTest)

# One binary per suite, so `ctest -R ImageOpsTest` targets one and a crash in
# one area cannot take the rest down.
set(TEST_SUITES
  ImageOpsTest.cpp
  CraftOpsTest.cpp
  CraftParityTest.cpp)

foreach(suite_path IN LISTS TEST_SUITES)
  get_filename_component(suite_name "${suite_path}" NAME_WE)
  add_executable(${suite_name} "${suite_path}" support/Assertions.h)
  target_include_directories(${suite_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(${suite_name} PRIVATE rnocr_kernels rnocr_reference_craft GTest::gtest_main)
  target_compile_definitions(${suite_name} PRIVATE RNOCR_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  target_compile_options(${suite_name} PRIVATE -Wall -Wextra)
  gtest_discover_tests(${suite_name} DISCOVERY_TIMEOUT 60)
endforeach()
