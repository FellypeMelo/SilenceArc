cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find oneAPI SYCL
find_package(IntelSYCL REQUIRED)

add_subdirectory(third_party/imgui)

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
    target_link_libraries(test_harness PRIVATE IntelSYCL::SYCL imgui)
else()
    target_include_directories(test_harness PRIVATE ${SYCL_INCLUDE_DIRS})
    target_link_libraries(test_harness PRIVATE ${SYCL_LIBRARIES} imgui)
endif()

# UI Manager
add_executable(test_ui_manager
    tests/test_ui_manager.cpp
    src/infrastructure/ui_manager.cpp
)
target_link_libraries(test_ui_manager PRIVATE imgui)

# DeepFilterNet integration (Stubbed for now)
# add_executable(test_noise_suppression
#     tests/test_noise_suppression.cpp
# )
# target_link_libraries(test_noise_suppression PRIVATE GTest::gtest_main)

enable_testing()
add_test(NAME BenchmarkHarnessTest COMMAND test_harness)
add_test(NAME UIManagerTest COMMAND test_ui_manager)
