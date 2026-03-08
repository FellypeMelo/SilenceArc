cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find oneAPI SYCL
find_package(IntelSYCL REQUIRED)

include_directories(include)

add_executable(test_harness 
    tests/test_benchmark_harness.cpp
    src/benchmark_harness.cpp
)

# Force -fsycl for both compilation and linking
target_compile_options(test_harness PRIVATE -fsycl)
target_link_options(test_harness PRIVATE -fsycl)

# Link using variables if target is missing
if(TARGET IntelSYCL::SYCL)
    target_link_libraries(test_harness PRIVATE IntelSYCL::SYCL)
else()
    target_include_directories(test_harness PRIVATE ${SYCL_INCLUDE_DIRS})
    target_link_libraries(test_harness PRIVATE ${SYCL_LIBRARIES})
endif()

enable_testing()
add_test(NAME BenchmarkHarnessTest COMMAND test_harness)
