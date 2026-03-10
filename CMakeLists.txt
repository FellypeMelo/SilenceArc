cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

# oneAPI SYCL configuration
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "[INFO] Intel LLVM compiler detected. Configuring SYCL.")
    add_compile_options("-fsycl")
    add_link_options("-fsycl")
    if(WIN32)
        add_compile_options("/EHsc") # Enable exceptions
        link_directories("C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    endif()
    
    # oneDNN (DNNL)
    find_package(DNNL REQUIRED)
    message(STATUS "[INFO] oneDNN found: ${DNNL_VERSION}")

    # oneMKL
    # Set MKL_ROOT if needed, but it should be in the environment from setvars.bat
    find_package(MKL REQUIRED)
    message(STATUS "[INFO] oneMKL found.")

    # Bypass broken IntelSYCLConfig.cmake if needed
    if(NOT TARGET IntelSYCL::SYCL)
        add_library(IntelSYCL::SYCL INTERFACE IMPORTED)
    endif()
    set(USE_GTEST OFF)
else()
    message(WARNING "[WARN] Not using Intel LLVM compiler. SYCL features may be disabled.")
    set(USE_GTEST ON)
endif()

# JSON Library
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_subdirectory(third_party/imgui)

include_directories(include)
include_directories(tests) # For sycl_test_harness.h
include_directories(third_party/miniaudio)

# GoogleTest (Optional)
if(USE_GTEST)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

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
    src/infrastructure/async_audio_pipeline.cpp
    src/infrastructure/miniaudio_device_manager.cpp
    src/infrastructure/miniaudio_pipeline.cpp
    src/infrastructure/sycl_accelerator.cpp
    src/infrastructure/onednn_inference_engine.cpp
)
target_link_libraries(silence_arc_infra PUBLIC imgui df_shared nlohmann_json::nlohmann_json)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # We need to link against the specific MKL SYCL libraries
    # MKL_SYCL_LIBRARIES is often defined by MKLConfig.cmake
    target_link_libraries(silence_arc_infra PUBLIC DNNL::dnnl MKL::MKL)
    
    # Try to add explicit SYCL DFT linkage if the meta-target is insufficient
    target_link_libraries(silence_arc_infra PUBLIC 
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_sycl_dft_dll.lib"
    )
endif()

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

if(USE_GTEST)
    # Noise Suppression Test (GTest)
    add_executable(test_noise_suppression
        tests/test_noise_suppression.cpp
    )
    target_link_libraries(test_noise_suppression PRIVATE 
        GTest::gtest_main 
        silence_arc_infra
    )

    # Audio Pipeline Test (GTest)
    add_executable(test_audio_pipeline
        tests/test_audio_pipeline.cpp
    )
    target_link_libraries(test_audio_pipeline PRIVATE 
        GTest::gtest_main 
        silence_arc_infra
    )

    # E2E Loopback Test (GTest)
    add_executable(test_e2e_loopback
        tests/test_e2e_loopback.cpp
    )
    target_link_libraries(test_e2e_loopback PRIVATE 
        GTest::gtest_main 
        silence_arc_infra
    )

    # E2E Samples Test (GTest)
    add_executable(test_e2e_samples
        tests/test_e2e_samples.cpp
    )
    target_include_directories(test_e2e_samples PRIVATE tests)
    target_link_libraries(test_e2e_samples PRIVATE 
        GTest::gtest_main 
        silence_arc_infra
    )

    # Audio Stream Buffer Test (GTest)
    add_executable(test_audio_stream_buffer
        tests/test_audio_stream_buffer.cpp
    )
    target_include_directories(test_audio_stream_buffer PRIVATE tests)
    target_link_libraries(test_audio_stream_buffer PRIVATE 
        GTest::gtest_main 
        silence_arc_infra
    )
endif()

# Main Application
add_executable(silence_arc src/main.cpp)
target_link_libraries(silence_arc PRIVATE silence_arc_infra)

enable_testing()
add_test(NAME BenchmarkHarnessTest COMMAND test_harness)
add_test(NAME UIManagerTest COMMAND test_ui_manager)

if(USE_GTEST)
    add_test(NAME NoiseSuppressionTest COMMAND test_noise_suppression)
    add_test(NAME AudioPipelineTest COMMAND test_audio_pipeline)
    add_test(NAME E2ELoopbackTest COMMAND test_e2e_loopback)
    add_test(NAME AudioStreamBufferTest COMMAND test_audio_stream_buffer)
endif()

# SYCL Discovery Test (Custom Harness)
add_executable(test_sycl_discovery
    tests/test_sycl_discovery.cpp
)
target_link_libraries(test_sycl_discovery PRIVATE IntelSYCL::SYCL)

# GPU Bridge FFI Test
add_executable(test_gpu_bridge
    tests/test_gpu_bridge.cpp
)
target_link_libraries(test_gpu_bridge PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Kernel Correctness Test
add_executable(test_kernel_correctness
    tests/test_kernel_correctness.cpp
)
target_link_libraries(test_kernel_correctness PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Neural Network Layers Test
add_executable(test_nn_layers
    tests/test_nn_layers.cpp
)
target_link_libraries(test_nn_layers PRIVATE IntelSYCL::SYCL silence_arc_infra)

add_test(NAME SYCLDiscoveryTest COMMAND test_sycl_discovery)
add_test(NAME GPUBridgeTest COMMAND test_gpu_bridge)
add_test(NAME KernelCorrectnessTest COMMAND test_kernel_correctness)
add_test(NAME NNLayersTest COMMAND test_nn_layers)
