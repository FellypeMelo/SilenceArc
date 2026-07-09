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
target_include_directories(imgui_lib SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC d3d11 dxgi)
# Vendored third party we do not maintain: silence its own compiler warnings so
# they do not drown out warnings from our code under -Wall/-Wextra.
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(imgui_lib PRIVATE -w)
endif()

include_directories(include)
include_directories(tests) # For sycl_test_harness.h
# miniaudio is a large single-header third party; treat as SYSTEM so -Wall/-Wextra
# do not flood us with warnings from code we do not own.
include_directories(SYSTEM third_party/miniaudio)

# Level Zero SDK (installed version varies by machine; auto-detect the newest
# version directory under the default install root instead of hardcoding one).
file(GLOB SILENCE_ARC_LEVEL_ZERO_CANDIDATES "C:/Program Files/LevelZeroSDK/*")
list(SORT SILENCE_ARC_LEVEL_ZERO_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
list(GET SILENCE_ARC_LEVEL_ZERO_CANDIDATES 0 SILENCE_ARC_LEVEL_ZERO_DEFAULT)
set(SILENCE_ARC_LEVEL_ZERO_ROOT "${SILENCE_ARC_LEVEL_ZERO_DEFAULT}" CACHE PATH "Level Zero SDK root (contains include/ and lib/)")

find_path(LEVEL_ZERO_INCLUDE_DIR NAMES level_zero/ze_api.h HINTS "${SILENCE_ARC_LEVEL_ZERO_ROOT}/include")
find_library(LEVEL_ZERO_LIB NAMES ze_loader HINTS "${SILENCE_ARC_LEVEL_ZERO_ROOT}/lib")
if(NOT LEVEL_ZERO_INCLUDE_DIR OR NOT LEVEL_ZERO_LIB)
    message(FATAL_ERROR "Level Zero SDK not found. Set SILENCE_ARC_LEVEL_ZERO_ROOT to its install root (containing include/ and lib/).")
endif()
message(STATUS "[INFO] Level Zero SDK: ${SILENCE_ARC_LEVEL_ZERO_ROOT}")
include_directories(SYSTEM ${LEVEL_ZERO_INCLUDE_DIR})
get_filename_component(LEVEL_ZERO_LIB_DIR "${LEVEL_ZERO_LIB}" DIRECTORY)
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

    # The MKL::MKL meta-target does not always pull in the SYCL DFT import lib we
    # need for oneapi::mkl::dft, so locate it explicitly via the MKLROOT the oneAPI
    # setvars environment exports (instead of a hardcoded absolute path).
    find_library(SILENCE_ARC_MKL_SYCL_DFT
        NAMES mkl_sycl_dft_dll mkl_sycl_dft
        HINTS "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64"
        DOC "oneMKL SYCL DFT import library")
    if(NOT SILENCE_ARC_MKL_SYCL_DFT)
        message(WARNING "mkl_sycl_dft_dll not found under MKLROOT ($ENV{MKLROOT}); SYCL DFT linkage may fail.")
    else()
        message(STATUS "[INFO] MKL SYCL DFT lib: ${SILENCE_ARC_MKL_SYCL_DFT}")
    endif()

    # Bypass broken IntelSYCLConfig.cmake if needed
    if(NOT TARGET IntelSYCL::SYCL)
        add_library(IntelSYCL::SYCL INTERFACE IMPORTED)
    endif()
endif()

# Applies the icx/DPC++ SYCL compile+link flags to a target. Replaces the
# identical -fsycl/EHsc/target-triple block that was duplicated across every
# SYCL executable. VIS is PUBLIC (for the infra lib, so consumers inherit) or
# PRIVATE (for leaf executables).
function(silence_arc_sycl_flags TARGET_NAME VIS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(${TARGET_NAME} ${VIS} -fsycl /EHsc --target=x86_64-pc-windows-msvc -Wno-ignored-attributes -Wall -Wextra)
        target_link_options(${TARGET_NAME} ${VIS} -fsycl --target=x86_64-pc-windows-msvc)
    endif()
endfunction()

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
    src/infrastructure/sycl_noise_suppressor.cpp
    src/infrastructure/onednn_inference_engine.cpp
    src/infrastructure/sycl_telemetry_provider.cpp
)
target_link_libraries(silence_arc_infra PUBLIC imgui_lib df_shared nlohmann_json::nlohmann_json ze_loader)

# Any target linking silence_arc_infra implicitly loads df.dll (via df_shared) at
# process start, so it must sit next to the produced executables, not just next
# to the source tree.
add_custom_command(TARGET silence_arc_infra POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/df.dll" "${CMAKE_BINARY_DIR}/df.dll"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    silence_arc_sycl_flags(silence_arc_infra PUBLIC)
    target_link_libraries(silence_arc_infra PUBLIC DNNL::dnnl MKL::MKL)
    if(SILENCE_ARC_MKL_SYCL_DFT)
        target_link_libraries(silence_arc_infra PUBLIC "${SILENCE_ARC_MKL_SYCL_DFT}")
    endif()
endif()

# test_harness (Simulation)
add_executable(test_harness
    tests/test_benchmark_harness.cpp
    src/benchmark_harness.cpp
)
silence_arc_sycl_flags(test_harness PRIVATE)

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
    add_test(NAME E2ESamplesTest COMMAND test_e2e_samples)
    add_test(NAME AudioStreamBufferTest COMMAND test_audio_stream_buffer)
    add_executable(test_telemetry_provider tests/test_telemetry_provider.cpp)
    target_link_libraries(test_telemetry_provider PRIVATE GTest::gtest_main silence_arc_infra)
    silence_arc_sycl_flags(test_telemetry_provider PRIVATE)
    add_test(NAME TelemetryProviderTest COMMAND test_telemetry_provider)

    # Backend parity: SYCL/oneDNN vs CPU/tract on real speech samples. Skips
    # gracefully when no SYCL device is present (CI runners without an Arc GPU).
    add_executable(test_backend_parity tests/test_backend_parity.cpp)
    target_link_libraries(test_backend_parity PRIVATE GTest::gtest_main silence_arc_infra)
    silence_arc_sycl_flags(test_backend_parity PRIVATE)
    add_test(NAME BackendParityTest COMMAND test_backend_parity)
endif()

# SYCL Discovery Test (Custom Harness)
add_executable(test_sycl_discovery
    tests/test_sycl_discovery.cpp
)
silence_arc_sycl_flags(test_sycl_discovery PRIVATE)
target_link_libraries(test_sycl_discovery PRIVATE IntelSYCL::SYCL)

# GPU Bridge FFI Test
add_executable(test_gpu_bridge
    tests/test_gpu_bridge.cpp
)
silence_arc_sycl_flags(test_gpu_bridge PRIVATE)
target_link_libraries(test_gpu_bridge PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Kernel Correctness Test
add_executable(test_kernel_correctness
    tests/test_kernel_correctness.cpp
)
silence_arc_sycl_flags(test_kernel_correctness PRIVATE)
target_link_libraries(test_kernel_correctness PRIVATE IntelSYCL::SYCL silence_arc_infra)

# Neural Network Layers Test
add_executable(test_nn_layers
    tests/test_nn_layers.cpp
)
silence_arc_sycl_flags(test_nn_layers PRIVATE)
target_link_libraries(test_nn_layers PRIVATE IntelSYCL::SYCL silence_arc_infra)

add_test(NAME SYCLDiscoveryTest COMMAND test_sycl_discovery)
add_test(NAME GPUBridgeTest COMMAND test_gpu_bridge)
add_test(NAME KernelCorrectnessTest COMMAND test_kernel_correctness)
add_test(NAME NNLayersTest COMMAND test_nn_layers)

# Real-pipeline latency benchmark (p50/p90/p99/max of sycl_process()).
add_executable(bench_pipeline_latency tests/bench_pipeline_latency.cpp)
silence_arc_sycl_flags(bench_pipeline_latency PRIVATE)
target_link_libraries(bench_pipeline_latency PRIVATE silence_arc_infra)
add_test(NAME PipelineLatencyBench COMMAND bench_pipeline_latency)
