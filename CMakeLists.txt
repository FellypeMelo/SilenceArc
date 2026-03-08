cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

# Find oneAPI SYCL
find_package(IntelSYCL REQUIRED)

add_subdirectory(third_party/imgui)

include_directories(include)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# DeepFilterNet Library (External Rust DLL)
add_library(df_shared SHARED IMPORTED)
set_target_properties(df_shared PROPERTIES
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/DeepFilterNet/target/release/df.dll.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/DeepFilterNet/target/release/df.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# SilenceArc Library (Infrastructure)
add_library(silence_arc_infra STATIC
    src/infrastructure/ui_manager.cpp
    src/infrastructure/deep_filter_adapter.cpp
)
target_link_libraries(silence_arc_infra PUBLIC imgui df_shared)

# test_harness (Simulation)
add_executable(test_harness 
    tests/test_benchmark_harness.cpp
    src/benchmark_harness.cpp
)
target_compile_options(test_harness PRIVATE -fsycl)
target_link_options(test_harness PRIVATE -fsycl)
if(TARGET IntelSYCL::SYCL)
    target_link_libraries(test_harness PRIVATE IntelSYCL::SYCL imgui)
else()
    target_include_directories(test_harness PRIVATE ${SYCL_INCLUDE_DIRS})
    target_link_libraries(test_harness PRIVATE ${SYCL_LIBRARIES} imgui)
endif()

# UI Manager Test
add_executable(test_ui_manager
    tests/test_ui_manager.cpp
)
target_link_libraries(test_ui_manager PRIVATE silence_arc_infra)

# Noise Suppression Test (GTest)
add_executable(test_noise_suppression
    tests/test_noise_suppression.cpp
)
target_link_libraries(test_noise_suppression PRIVATE 
    GTest::gtest_main 
    silence_arc_infra
)

enable_testing()
add_test(NAME BenchmarkHarnessTest COMMAND test_harness)
add_test(NAME UIManagerTest COMMAND test_ui_manager)
add_test(NAME NoiseSuppressionTest COMMAND test_noise_suppression)
