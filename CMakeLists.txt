cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
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
include_directories(tests) 
include_directories(tests/utils)
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
    
    # Correctly add GTest includes after FetchContent
    include_directories(${googletest_SOURCE_DIR}/googletest/include)
    include_directories(${googletest_SOURCE_DIR}/googlemock/include)
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
    src/infrastructure/sycl_telemetry_provider.cpp
    src/infrastructure/sycl/sycl_memory_manager.cpp
    src/infrastructure/sycl/sycl_dsp_engine.cpp
    src/infrastructure/sycl/sycl_graph_builder.cpp
    src/infrastructure/sycl/linear_layer.cpp
    src/infrastructure/sycl/grouped_linear_layer.cpp
    src/infrastructure/sycl/binary_add_layer.cpp
    src/infrastructure/sycl/conv2d_layer.cpp
    src/infrastructure/sycl/normalization_layer.cpp
    src/infrastructure/sycl/native_sycl_engine.cpp
    src/domain/audio_stream_buffer.cpp
)
target_link_libraries(silence_arc_infra PUBLIC imgui df_shared nlohmann_json::nlohmann_json ze_loader)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # Crucial flags for icx on Windows with SYCL
    target_compile_options(silence_arc_infra PUBLIC -fsycl /EHsc --target=x86_64-pc-windows-msvc -Wno-ignored-attributes)
    target_link_options(silence_arc_infra PUBLIC -fsycl --target=x86_64-pc-windows-msvc)

    # We need to link against the specific MKL SYCL libraries
    target_link_libraries(silence_arc_infra PUBLIC DNNL::dnnl MKL::MKL)

    # Try to add explicit SYCL DFT linkage if the meta-target is insufficient
    # Explicitly add mkl_sycl_dft and core libraries
    target_link_libraries(silence_arc_infra PUBLIC
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_sycl_dft_dll.lib"
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_intel_ilp64_dll.lib"
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_core_dll.lib"
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_intel_thread_dll.lib"
    )
endif()

# UI Manager Test
add_executable(test_ui_manager
    tests/test_ui_manager.cpp
)
target_link_libraries(test_ui_manager PRIVATE GTest::gtest_main silence_arc_infra)
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(test_ui_manager PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
    target_link_options(test_ui_manager PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
endif()

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

    # SYCL Foundation Test (GTest)
    add_executable(test_sycl_foundation
        tests/test_sycl_foundation.cpp
    )
    target_link_libraries(test_sycl_foundation PRIVATE
        GTest::gtest_main
        silence_arc_infra
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(test_sycl_foundation PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
        target_link_options(test_sycl_foundation PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
    endif()
    add_test(NAME SyclFoundationTest COMMAND test_sycl_foundation)

    # SYCL DSP Test (GTest)
    add_executable(test_sycl_dsp
        tests/test_sycl_dsp.cpp
    )
    target_link_libraries(test_sycl_dsp PRIVATE
        GTest::gtest_main
        silence_arc_infra
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(test_sycl_dsp PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
        target_link_options(test_sycl_dsp PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
    endif()
    add_test(NAME SyclDspTest COMMAND test_sycl_dsp)

    # SYCL Graph Test (GTest)
    add_executable(test_sycl_graph
        tests/test_sycl_graph.cpp
    )
    target_link_libraries(test_sycl_graph PRIVATE
        GTest::gtest_main
        silence_arc_infra
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(test_sycl_graph PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
        target_link_options(test_sycl_graph PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
    endif()
    add_test(NAME SyclGraphTest COMMAND test_sycl_graph)

    # Native SYCL Engine Test (GTest)
    add_executable(test_native_sycl_engine
        tests/test_native_sycl_engine.cpp
    )
    target_link_libraries(test_native_sycl_engine PRIVATE
        GTest::gtest_main
        silence_arc_infra
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        target_compile_options(test_native_sycl_engine PRIVATE -fsycl /EHsc --target=x86_64-pc-windows-msvc)
        target_link_options(test_native_sycl_engine PRIVATE -fsycl --target=x86_64-pc-windows-msvc)
    endif()
    add_test(NAME NativeSyclEngineTest COMMAND test_native_sycl_engine)
endif()
