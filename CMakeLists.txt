cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

# Dependencies
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.1
)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(json imgui glfw)

# ImGui setup
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC d3d11 dxgi)

include_directories(include)
include_directories(tests) # For sycl_test_harness.h
include_directories(third_party/miniaudio)

# Level Zero SDK Paths (Detected)
set(LEVEL_ZERO_INCLUDE_DIR "C:/Program Files/LevelZeroSDK/1.26.1/include")
set(LEVEL_ZERO_LIB_DIR "C:/Program Files/LevelZeroSDK/1.26.1/lib")
include_directories(${LEVEL_ZERO_INCLUDE_DIR})
link_directories(${LEVEL_ZERO_LIB_DIR})

# GoogleTest (Optional)
set(USE_GTEST ON)
if(USE_GTEST)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# oneAPI SYCL configuration
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "[INFO] Intel LLVM compiler detected. Configuring SYCL targets.")
    
    if(WIN32)
        link_directories("C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    endif()

    # oneDNN (DNNL)
    find_package(DNNL REQUIRED)
    message(STATUS "[INFO] oneDNN found: ${DNNL_VERSION}")

    # oneMKL
    find_package(MKL REQUIRED)
    message(STATUS "[INFO] oneMKL found.")

    # Bypass broken IntelSYCLConfig.cmake if needed
    if(NOT TARGET IntelSYCL::SYCL)
        add_library(IntelSYCL::SYCL INTERFACE IMPORTED)
    endif()
endif()

# DeepFilterNet Library (Rust DLL)
add_library(df_shared SHARED IMPORTED)
set_target_properties(df_shared PROPERTIES
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/df.dll.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/df.dll"
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
    src/infrastructure/sycl_telemetry_provider.cpp
)
target_link_libraries(silence_arc_infra PUBLIC imgui_lib df_shared nlohmann_json::nlohmann_json ze_loader)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # Crucial flags for icx on Windows with SYCL
    target_compile_options(silence_arc_infra PUBLIC -fsycl /EHsc --target=x86_64-pc-windows-msvc -Wno-ignored-attributes)
    target_link_options(silence_arc_infra PUBLIC -fsycl --target=x86_64-pc-windows-msvc)
    
    # We need to link against the specific MKL SYCL libraries
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
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_harness PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_harness PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()

if(TARGET IntelSYCL::SYCL)
    target_link_libraries(test_harness PRIVATE IntelSYCL::SYCL imgui_lib)
else()
    target_include_directories(test_harness PRIVATE ${SYCL_INCLUDE_DIRS})
    target_link_libraries(test_harness PRIVATE ${SYCL_LIBRARIES} imgui_lib)
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
    add_executable(test_telemetry_provider tests/test_telemetry_provider.cpp)
    target_link_libraries(test_telemetry_provider PRIVATE GTest::gtest_main silence_arc_infra)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(test_telemetry_provider PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
        target_link_options(test_telemetry_provider PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
    endif()
    add_test(NAME TelemetryProviderTest COMMAND test_telemetry_provider)
endif()

# SYCL Discovery Test (Custom Harness)
add_executable(test_sycl_discovery
    tests/test_sycl_discovery.cpp
)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_sycl_discovery PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_sycl_discovery PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()
target_link_libraries(test_sycl_discovery PRIVATE IntelSYCL::SYCL)

# GPU Bridge FFI Test
add_executable(test_gpu_bridge
    tests/test_gpu_bridge.cpp
)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_gpu_bridge PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_gpu_bridge PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()
target_link_libraries(test_gpu_bridge PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Kernel Correctness Test
add_executable(test_kernel_correctness
    tests/test_kernel_correctness.cpp
)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_kernel_correctness PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_kernel_correctness PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()
target_link_libraries(test_kernel_correctness PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Neural Network Layers Test
add_executable(test_nn_layers
    tests/test_nn_layers.cpp
)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_nn_layers PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_nn_layers PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()
target_link_libraries(test_nn_layers PRIVATE IntelSYCL::SYCL silence_arc_infra)

add_test(NAME SYCLDiscoveryTest COMMAND test_sycl_discovery)
add_test(NAME GPUBridgeTest COMMAND test_gpu_bridge)
add_test(NAME KernelCorrectnessTest COMMAND test_kernel_correctness)
add_test(NAME NNLayersTest COMMAND test_nn_layers)
