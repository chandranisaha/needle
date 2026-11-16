cmake_minimum_required(VERSION 3.20)
project(needle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NEEDLE_NATIVE "Tune code for the machine doing the build" ON)
option(NEEDLE_FORCE_SCALAR "Use the scalar distance kernel even where AVX2 is available" OFF)

find_package(Threads REQUIRED)

add_library(needle
  src/dataset.cpp
  src/brute_force.cpp
  src/vamana.cpp)
target_include_directories(needle PUBLIC include)
target_link_libraries(needle PUBLIC Threads::Threads)
if(NEEDLE_FORCE_SCALAR)
  target_compile_definitions(needle PUBLIC NEEDLE_FORCE_SCALAR)
endif()
if(NOT MSVC)
  target_compile_options(needle PUBLIC -Wall -Wextra -Wpedantic)
  if(NEEDLE_NATIVE)
    target_compile_options(needle PUBLIC -march=native)
  endif()
endif()

add_executable(needle_bench bench/bench_main.cpp)
target_link_libraries(needle_bench PRIVATE needle)

add_executable(needle_unreachable tools/unreachable.cpp)
target_link_libraries(needle_unreachable PRIVATE needle)

add_executable(needle_export_demo tools/export_demo.cpp)
target_link_libraries(needle_export_demo PRIVATE needle)

add_executable(needle_export_words tools/export_words.cpp)
target_link_libraries(needle_export_words PRIVATE needle)

# Baseline: git clone https://github.com/nmslib/hnswlib third_party/hnswlib
set(NEEDLE_HNSWLIB_DIR "${CMAKE_SOURCE_DIR}/third_party/hnswlib" CACHE PATH "hnswlib checkout")
if(EXISTS "${NEEDLE_HNSWLIB_DIR}/hnswlib/hnswlib.h")
  add_executable(needle_hnsw_baseline bench/hnsw_baseline.cpp)
  target_include_directories(needle_hnsw_baseline SYSTEM PRIVATE "${NEEDLE_HNSWLIB_DIR}")
  target_link_libraries(needle_hnsw_baseline PRIVATE needle)
endif()

enable_testing()
add_executable(needle_tests tests/test_main.cpp)
target_link_libraries(needle_tests PRIVATE needle)
add_test(NAME needle_tests COMMAND needle_tests)
